use crate::{
    EditPredictionId, EditPredictionModelInput, cursor_excerpt,
    prediction::EditPredictionResult,
    zeta1::{
        self, MAX_CONTEXT_TOKENS as ZETA_MAX_CONTEXT_TOKENS,
        MAX_EVENT_TOKENS as ZETA_MAX_EVENT_TOKENS,
    },
};
use anyhow::{Context as _, Result};
use cloud_llm_client::predict_edits_v3::RawCompletionRequest;
use futures::AsyncReadExt as _;
use gpui::{App, AppContext as _, Entity, SharedString, Task, http_client};
use language::{
    Anchor, Buffer, BufferSnapshot, OffsetRangeExt as _, ToOffset, ToPoint as _,
    language_settings::all_language_settings,
};
use language_model::{LanguageModelProviderId, LanguageModelRegistry};
use serde::{Deserialize, Serialize};
use settings::{EditPredictionPromptFormat, EditPredictionProvider};
use std::{path::Path, sync::Arc, time::Instant};
use zeta_prompt::{
    ZetaPromptInput,
    zeta1::{EDITABLE_REGION_END_MARKER, format_zeta1_prompt},
};

const FIM_CONTEXT_TOKENS: usize = 512;

pub fn request_prediction(
    &self,
    prompt_format: EditPredictionPromptFormat,
    EditPredictionModelInput {
        buffer,
        snapshot,
        position,
        events,
        ..
    }: EditPredictionModelInput,
    cx: &mut App,
) -> Task<Result<Option<EditPredictionResult>>> {
    let settings = all_language_settings(None, cx).edit_predictions.clone();

    log::debug!("Ollama: Requesting completion (model: {})", model);

    let full_path: Arc<Path> = snapshot
        .file()
        .map(|file| file.full_path(cx))
        .unwrap_or_else(|| "untitled".into())
        .into();

    let http_client = cx.http_client();
    let cursor_point = position.to_point(&snapshot);
    let buffer_snapshotted_at = Instant::now();

    // Zeta generates more tokens than FIM models. Ideally, we'd use MAX_REWRITE_TOKENS,
    // but this might be too slow for local deployments. So we make it configurable,
    // but we also have this hardcoded multiplier for now.

    let result = cx.background_spawn(async move {
        // For zeta models, use the dedicated zeta1 functions which handle their own
        // range computation with the correct token limits.
        let (excerpt_range, _) = cursor_excerpt::editable_and_context_ranges_for_cursor_position(
            cursor_point,
            &snapshot,
            FIM_CONTEXT_TOKENS,
            0,
        );
        let excerpt_offset_range = excerpt_range.to_offset(&snapshot);
        let cursor_offset = cursor_point.to_offset(&snapshot);

        let inputs = ZetaPromptInput {
            events,
            related_files: Vec::new(),
            cursor_offset_in_excerpt: cursor_offset - excerpt_offset_range.start,
            editable_range_in_excerpt: cursor_offset - excerpt_offset_range.start
                ..cursor_offset - excerpt_offset_range.start,
            cursor_path: full_path.clone(),
            excerpt_start_row: Some(excerpt_range.start.row),
            cursor_excerpt: snapshot
                .text_for_range(excerpt_range)
                .collect::<String>()
                .into(),
            excerpt_ranges: None,
            preferred_model: None,
            in_open_source_repo: false,
            can_collect_data: false,
        };

        let prefix = inputs.cursor_excerpt[..inputs.cursor_offset_in_excerpt].to_string();
        let suffix = inputs.cursor_excerpt[inputs.cursor_offset_in_excerpt..].to_string();
        let prompt = format_fim_prompt(&model, &prefix, &suffix);
        let stop_tokens = get_fim_stop_tokens();

        let response_text = match settings.provider {
            EditPredictionProvider::Ollama => {
                let request = OllamaGenerateRequest {
                    model: model.clone(),
                    prompt,
                    raw: true,
                    stream: false,
                    options: Some(OllamaGenerateOptions {
                        num_predict: Some(max_output_tokens),
                        temperature: Some(0.2),
                        stop: Some(stop_tokens),
                    }),
                };

                let request_body = serde_json::to_string(&request)?;
                let http_request = http_client::Request::builder()
                    .method(http_client::Method::POST)
                    .uri(format!("{}/api/generate", settings.ollama.api_url))
                    .header("Content-Type", "application/json")
                    .body(http_client::AsyncBody::from(request_body))?;

                let mut response = http_client.send(http_request).await?;
                let status = response.status();

                log::debug!("Ollama: Response status: {}", status);

                if !status.is_success() {
                    let mut body = String::new();
                    response.body_mut().read_to_string(&mut body).await?;
                    return Err(anyhow::anyhow!("Ollama API error: {} - {}", status, body));
                }

                let mut body = String::new();
                response.body_mut().read_to_string(&mut body).await?;

                let ollama_response: OllamaGenerateResponse =
                    serde_json::from_str(&body).context("Failed to parse Ollama response")?;
                ollama_response.response
            }
            EditPredictionProvider::OpenAiCompatibleApi => {
                let request = RawCompletionRequest {
                    model: model.clone(),
                    prompt,
                    max_tokens: settings.openai_compatible.max_output_tokens,
                    stream: false,
                };
            }
            _ => {}
        };

        let response_received_at = Instant::now();

        log::debug!(
            "Ollama: Completion received ({:.2}s)",
            (response_received_at - buffer_snapshotted_at).as_secs_f64()
        );

        let completion: Arc<str> = clean_fim_completion(&ollama_response.response).into();
        let edits = if completion.is_empty() {
            vec![]
        } else {
            let cursor_offset = cursor_point.to_offset(&snapshot);
            let anchor = snapshot.anchor_after(cursor_offset);
            vec![(anchor..anchor, completion)]
        };

        anyhow::Ok(OllamaRequestOutput {
            created_at: ollama_response.created_at,
            edits,
            snapshot,
            response_received_at,
            inputs,
            buffer,
            buffer_snapshotted_at,
        })
    });

    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        let output = result.await.context("Ollama edit prediction failed")?;
        anyhow::Ok(Some(
            EditPredictionResult::new(
                EditPredictionId(output.created_at.into()),
                &output.buffer,
                &output.snapshot,
                output.edits.into(),
                None,
                output.buffer_snapshotted_at,
                output.response_received_at,
                output.inputs,
                cx,
            )
            .await,
        ))
    })
}

fn is_zeta_model(model: &str) -> bool {
    model.to_lowercase().contains("zeta")
}

fn get_zeta_stop_tokens() -> Vec<String> {
    vec![EDITABLE_REGION_END_MARKER.to_string(), "```".to_string()]
}

fn format_fim_prompt(model: &str, prefix: &str, suffix: &str) -> String {
    let model_base = model.split(':').next().unwrap_or(model);

    match model_base {
        "codellama" | "code-llama" => {
            format!("<PRE> {prefix} <SUF>{suffix} <MID>")
        }
        "starcoder" | "starcoder2" | "starcoderbase" => {
            format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>")
        }
        "deepseek-coder" | "deepseek-coder-v2" => {
            format!("<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>")
        }
        "qwen2.5-coder" | "qwen-coder" | "qwen" => {
            format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
        }
        "codegemma" => {
            format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
        }
        "codestral" | "mistral" => {
            format!("[SUFFIX]{suffix}[PREFIX]{prefix}")
        }
        "glm" | "glm-4" | "glm-4.5" => {
            format!("<|code_prefix|>{prefix}<|code_suffix|>{suffix}<|code_middle|>")
        }
        _ => {
            format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>")
        }
    }
}

fn get_fim_stop_tokens() -> Vec<String> {
    vec![
        "<|endoftext|>".to_string(),
        "<|file_separator|>".to_string(),
        "<|fim_pad|>".to_string(),
        "<|fim_prefix|>".to_string(),
        "<|fim_middle|>".to_string(),
        "<|fim_suffix|>".to_string(),
        "<fim_prefix>".to_string(),
        "<fim_middle>".to_string(),
        "<fim_suffix>".to_string(),
        "<PRE>".to_string(),
        "<SUF>".to_string(),
        "<MID>".to_string(),
        "[PREFIX]".to_string(),
        "[SUFFIX]".to_string(),
    ]
}

fn clean_zeta_completion(mut response: &str) -> &str {
    if let Some(last_newline_ix) = response.rfind('\n') {
        let last_line = &response[last_newline_ix + 1..];
        if EDITABLE_REGION_END_MARKER.starts_with(&last_line) {
            response = &response[..last_newline_ix]
        }
    }
    response
}

fn clean_fim_completion(response: &str) -> String {
    let mut result = response.to_string();

    let end_tokens = [
        "<|endoftext|>",
        "<|file_separator|>",
        "<|fim_pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<fim_prefix>",
        "<fim_middle>",
        "<fim_suffix>",
        "<PRE>",
        "<SUF>",
        "<MID>",
        "[PREFIX]",
        "[SUFFIX]",
    ];

    for token in &end_tokens {
        if let Some(pos) = result.find(token) {
            result.truncate(pos);
        }
    }

    result
}
