import { useEffect, useMemo, useState } from "react";
import {
  b64ToDataUrlPng,
  fetchModels,
  predict,
  type PredictResponse,
} from "./lib/api";

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [consentStore, setConsentStore] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m);
        setModel(m[0] ?? "");
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const overlayUrl = useMemo(
    () => b64ToDataUrlPng(result?.gradcam?.overlay_png_b64),
    [result]
  );
  const heatmapUrl = useMemo(
    () => b64ToDataUrlPng(result?.gradcam?.heatmap_png_b64),
    [result]
  );

  async function onRun() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await predict(file, model, consentStore);
      setResult(r);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="h1">Bone Fracture Assessment (Explainable AI)</div>
        <div className="muted">
          Upload an X-ray image for fracture vs non-fracture prediction with
          Grad-CAM justification.
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          <strong>Disclaimer:</strong> This system is a research-based decision
          support tool and not a replacement for professional medical diagnosis.
        </div>
      </div>

      {error && (
        <div className="error" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}

      <div className="row">
        <div className="card">
          <div className="label">Model</div>
          <select
            className="select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          <div style={{ height: 12 }} />

          <div className="label">Upload X-ray (JPG/PNG)</div>
          <input
            className="input"
            type="file"
            accept="image/png,image/jpeg"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />

          <div style={{ height: 12 }} />

          <label
            className="muted"
            style={{ display: "flex", gap: 8, alignItems: "center" }}
          >
            <input
              type="checkbox"
              checked={consentStore}
              onChange={(e) => setConsentStore(e.target.checked)}
            />
            I consent to storing this image for research/debugging (optional)
          </label>

          <div style={{ height: 12 }} />

          <button
            className="button"
            disabled={!file || !model || loading}
            onClick={onRun}
          >
            {loading ? "Running…" : "Run Assessment"}
          </button>

          <div style={{ height: 14 }} />

          <div className="muted">
            No images are stored unless you explicitly consent.
          </div>
        </div>

        <div className="card">
          <div className="h1">Results</div>

          {!result && (
            <div className="muted">Upload an image and run assessment.</div>
          )}

          {result && (
            <>
              {!result.valid && (
                <div className="error">
                  {result.validation?.message}
                  {result.validation?.reasons?.length ? (
                    <ul>
                      {result.validation.reasons.map((r, idx) => (
                        <li key={idx}>{r}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              )}

              {result.valid && (
                <>
                  <div className="kv">
                    <div>Prediction</div>
                    <div>
                      <span className="badge">{result.prediction}</span>
                    </div>
                  </div>
                  <div className="kv">
                    <div>Confidence</div>
                    <div>
                      {result.confidence
                        ? `${(result.confidence * 100).toFixed(1)}%`
                        : "—"}
                    </div>
                  </div>
                  <div className="kv">
                    <div>Model</div>
                    <div>{result.model}</div>
                  </div>

                  <hr />

                  {result.warnings?.length ? (
                    <div className="warning">
                      {result.warnings.map((w, idx) => (
                        <div key={idx}>{w}</div>
                      ))}
                    </div>
                  ) : null}

                  <div style={{ height: 12 }} />

                  <div className="label">Explanation</div>
                  <div className="muted" style={{ fontSize: 13 }}>
                    {result.explanation ?? "—"}
                  </div>

                  <div style={{ height: 12 }} />

                  <div className="label">Grad-CAM</div>
                  <label
                    className="muted"
                    style={{
                      display: "flex",
                      gap: 8,
                      alignItems: "center",
                      marginBottom: 8,
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={showOverlay}
                      onChange={(e) => setShowOverlay(e.target.checked)}
                    />
                    Show heatmap overlay
                  </label>

                  <div className="muted" style={{ marginBottom: 8 }}>
                    Status: {result.gradcam?.status ?? "—"}{" "}
                    {result.gradcam?.message
                      ? `(${result.gradcam.message})`
                      : ""}
                  </div>

                  <div className="split">
                    <div>
                      <div className="label">Original</div>
                      <div className="imgBox">
                        {previewUrl ? (
                          <img src={previewUrl} alt="Uploaded X-ray" />
                        ) : null}
                      </div>
                    </div>
                    <div>
                      <div className="label">Heatmap</div>
                      <div className="imgBox">
                        {showOverlay && overlayUrl ? (
                          <img src={overlayUrl} alt="Grad-CAM overlay" />
                        ) : heatmapUrl ? (
                          <img src={heatmapUrl} alt="Grad-CAM heatmap" />
                        ) : (
                          <div className="muted" style={{ padding: 12 }}>
                            No heatmap available.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
