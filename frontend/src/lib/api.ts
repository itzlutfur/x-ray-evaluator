export type PredictResponse = {
  model: string;
  valid: boolean;
  validation: {
    message: string;
    reasons: string[];
    metrics: Record<string, unknown>;
  };
  prediction?: "Fracture" | "Non-Fracture";
  confidence?: number;
  warnings?: string[];
  explanation?: string;
  gradcam?: {
    heatmap_png_b64: string | null;
    overlay_png_b64: string | null;
    status?: string;
    message?: string | null;
  };
  disclaimer?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/v1/inference/models`);
  if (!res.ok) throw new Error("Failed to load models");
  const json = await res.json();
  return json.models as string[];
}

export async function predict(
  file: File,
  modelName: string,
  consentStore: boolean
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);
  form.append("consent_store", consentStore ? "true" : "false");

  const res = await fetch(`${API_BASE}/api/v1/inference/predict`, {
    method: "POST",
    body: form,
  });

  const json = (await res.json()) as PredictResponse | { detail: string };
  if (!res.ok) {
    const detail = (json as any).detail ?? "Request failed";
    throw new Error(detail);
  }
  return json as PredictResponse;
}

export function b64ToDataUrlPng(b64: string | null | undefined): string | null {
  if (!b64) return null;
  return `data:image/png;base64,${b64}`;
}
