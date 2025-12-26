import { DragEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  b64ToDataUrlPng,
  fetchModels,
  predict,
  type PredictResponse,
} from "./lib/api";

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [multiResults, setMultiResults] = useState<PredictResponse[]>([]);
  const [showOverlay, setShowOverlay] = useState(true);
  const [consentStore, setConsentStore] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [viewMode, setViewMode] = useState<"single" | "multi">("single");
  const [loadingAll, setLoadingAll] = useState(false);
  const [showResultsModal, setShowResultsModal] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((available) => {
        setModels(available);
        setModel(available[0] ?? "");
      })
      .catch((err) => setError(String(err?.message ?? err)));
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
  const hydratedMultiResults = useMemo(
    () =>
      multiResults.map((entry) => ({
        result: entry,
        overlayUrl: b64ToDataUrlPng(entry.gradcam?.overlay_png_b64),
        heatmapUrl: b64ToDataUrlPng(entry.gradcam?.heatmap_png_b64),
      })),
    [multiResults]
  );
  const fileSizeLabel = useMemo(() => {
    if (!file) return null;
    if (file.size < 1024) return `${file.size} B`;
    if (file.size < 1024 * 1024) return `${(file.size / 1024).toFixed(1)} kB`;
    return `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  }, [file]);
  const fileTypeLabel = useMemo(() => {
    if (!file) return null;
    return file.type?.trim() ? file.type : null;
  }, [file]);
  const fileModifiedLabel = useMemo(() => {
    if (!file) return null;
    const date = new Date(file.lastModified);
    if (Number.isNaN(date.getTime())) return null;
    return date.toLocaleString();
  }, [file]);

  useEffect(() => {
    if (!showResultsModal) {
      return;
    }
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [showResultsModal]);

  const handleFile = useCallback((next: File | null) => {
    setFile(next);
    setResult(null);
    setMultiResults([]);
    setShowResultsModal(false);
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setDragging(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) {
        handleFile(dropped);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  async function runInference() {
    if (!file || !model) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setMultiResults([]);
    setShowResultsModal(false);
    try {
      const response = await predict(file, model, consentStore);
      setResult(response);
      setShowResultsModal(true);
    } catch (err: any) {
      setError(String(err?.message ?? err));
      setShowResultsModal(false);
    } finally {
      setLoading(false);
    }
  }

  async function runInferenceAll() {
    if (!file || !models.length) return;
    setLoadingAll(true);
    setError(null);
    setResult(null);
    setMultiResults([]);
    setShowResultsModal(false);
    try {
      const responses: PredictResponse[] = [];
      for (const entry of models) {
        const response = await predict(file, entry, consentStore);
        responses.push(response);
      }
      setMultiResults(responses);
      setShowResultsModal(true);
    } catch (err: any) {
      setError(String(err?.message ?? err));
      setShowResultsModal(false);
    } finally {
      setLoadingAll(false);
    }
  }

  const isBusy = loading || loadingAll;
  const disableSingleRun = !file || !model || loading;
  const disableMultiRun = !file || !models.length || loadingAll;
  const canShowResults =
    viewMode === "single" ? Boolean(result) : multiResults.length > 0;
  const shouldRenderModal = showResultsModal && canShowResults;

  function changeMode(next: "single" | "multi") {
    if (viewMode === next) return;
    setViewMode(next);
    setError(null);
    setResult(null);
    setMultiResults([]);
    setShowResultsModal(false);
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <div className="heroEyebrow">Feature-focused preprocessing & XAI</div>
          <h1 className="heroTitle">Bone Fracture Assessment</h1>
          <p className="heroLead">
            <strong>Need a quick validation check?</strong> Run Grad-CAM
            overlays before accepting any prediction.
          </p>
        </div>
        <div className="heroBadges">
          <span className="pill">Grad-CAM overlay</span>
          <span className="pill">Validation-first pipeline</span>
          <span className="pill">No retraining required</span>
        </div>
      </header>

      <section className="notice">
        <strong>Disclaimer:</strong> Research-only decision support. Do not use
        for clinical diagnosis.
      </section>

      {error ? <div className="alert alert--error">{error}</div> : null}

      <main className="layout layout--split">
        <section className="panel panel--primary" id="upload-panel">
          <h2 className="panelTitle">Upload & Settings</h2>

          <div className="modeTabs" role="tablist" aria-label="Assessment mode">
            <button
              type="button"
              className={`modeTab ${
                viewMode === "single" ? "modeTab--active" : ""
              }`}
              onClick={() => changeMode("single")}
            >
              Single model
            </button>
            <button
              type="button"
              className={`modeTab ${
                viewMode === "multi" ? "modeTab--active" : ""
              }`}
              onClick={() => changeMode("multi")}
            >
              All models
            </button>
          </div>

          {viewMode === "single" ? (
            <>
              <label className="label" htmlFor="model">
                Choose model
              </label>
              <select
                id="model"
                className="select"
                value={model}
                onChange={(event) => {
                  setModel(event.target.value);
                  setResult(null);
                  setMultiResults([]);
                  setShowResultsModal(false);
                }}
              >
                {models.map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <p className="helper">
              Run the uploaded X-ray through every available backbone to compare
              predictions, confidence, and Grad-CAM overlays side by side.
            </p>
          )}

          <div className="label sectionLabel">Upload X-ray (JPG/PNG)</div>
          <label
            htmlFor="upload"
            className={`dropzone ${dragging ? "dropzone--active" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              id="upload"
              type="file"
              accept="image/png,image/jpeg"
              hidden
              onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
            />
            <div className="dropzoneIcon" aria-hidden>
              +
            </div>
            <div className="dropzoneText">
              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <span>Drag a different image to replace this file</span>
                </>
              ) : (
                <>
                  <strong>Drag & drop to upload</strong>
                  <span>or click to select from your device</span>
                </>
              )}
            </div>
          </label>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={consentStore}
              onChange={(event) => setConsentStore(event.target.checked)}
            />
            Store this upload for research/debugging (optional consent)
          </label>

          <div className="actions">
            {viewMode === "single" ? (
              <button
                className="button"
                type="button"
                disabled={disableSingleRun}
                onClick={runInference}
              >
                {loading ? "Running assessment…" : "Run assessment"}
              </button>
            ) : (
              <button
                className="button"
                type="button"
                disabled={disableMultiRun}
                onClick={runInferenceAll}
              >
                {loadingAll ? "Assessing all models…" : "Assess all models"}
              </button>
            )}
            <button
              className="button button--ghost"
              type="button"
              disabled={!file || isBusy}
              onClick={() => handleFile(null)}
            >
              Clear file
            </button>
            {canShowResults ? (
              <button
                className="button button--ghost"
                type="button"
                onClick={() => setShowResultsModal(true)}
              >
                View results
              </button>
            ) : null}
          </div>

          <p className="helper">
            No images persist unless consent is granted. Validation rejects
            low-quality or non X-ray inputs before inference runs.
          </p>
        </section>

        <section className="panel panel--neutral">
          <h2 className="panelTitle">Preview</h2>
          {previewUrl ? (
            <div className="preview">
              <div className="previewImage">
                <img src={previewUrl} alt="Uploaded preview" />
              </div>
              <div className="infoGrid">
                <div className="infoTile">
                  <div className="infoLabel">File name</div>
                  <div className="infoValue">{file?.name ?? "—"}</div>
                  <div className="infoHint">
                    Stored in memory only unless consent is granted.
                  </div>
                </div>
                <div className="infoTile">
                  <div className="infoLabel">Size</div>
                  <div className="infoValue">{fileSizeLabel ?? "—"}</div>
                  <div className="infoHint">
                    Large files are downscaled to 224×224 before inference.
                  </div>
                </div>
                <div className="infoTile">
                  <div className="infoLabel">Type</div>
                  <div className="infoValue">{fileTypeLabel ?? "—"}</div>
                  <div className="infoHint">
                    JPEG or PNG recommended for the preprocessing pipeline.
                  </div>
                </div>
                <div className="infoTile">
                  <div className="infoLabel">Last modified</div>
                  <div className="infoValue">{fileModifiedLabel ?? "—"}</div>
                  <div className="infoHint">
                    Browser metadata used for audit trails during demos.
                  </div>
                </div>
                <div className="infoTile">
                  <div className="infoLabel">Selected model</div>
                  <div className="infoValue">{model || "—"}</div>
                  <div className="infoHint">
                    Choose alternative backbones from the dropdown.
                  </div>
                </div>
                <div className="infoTile">
                  <div className="infoLabel">Consent</div>
                  <div className="infoValue">
                    {consentStore ? "Research storage permitted" : "No storage"}
                  </div>
                  <div className="infoHint">
                    Toggle the checkbox above before running assessment.
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <p className="helper">
              No image selected yet. Drag an X-ray to begin and the preview will
              appear here.
            </p>
          )}
        </section>
      </main>

      {shouldRenderModal ? (
        <div
          className="modal modal--open"
          role="dialog"
          aria-modal="true"
          aria-label="Assessment results"
        >
          <button
            type="button"
            className="modalBackdrop"
            onClick={() => setShowResultsModal(false)}
            aria-label="Close results"
          />
          <div className="modalContainer">
            <header className="modalHeader">
              <div>
                <div className="modalEyebrow">
                  {viewMode === "single" ? "Single model" : "All models"}
                </div>
                <h2 className="modalTitle">Assessment results</h2>
              </div>
              <button
                type="button"
                className="modalClose"
                onClick={() => setShowResultsModal(false)}
              >
                Close
              </button>
            </header>
            <div className="modalBody">
              {viewMode === "single" ? (
                result?.valid || result ? (
                  <>
                    {result && result.valid ? null : null}
                    {result && result.valid ? (
                      <div className="summaryGrid">
                        <div className="summaryTile">
                          <div className="summaryLabel">Prediction</div>
                          <div className="summaryValue">
                            <span className={`tag tag--${result.prediction}`}>
                              {result.prediction}
                            </span>
                          </div>
                          <div className="summaryHint">
                            Based on thesis-ready preprocessing.
                          </div>
                        </div>
                        <div className="summaryTile">
                          <div className="summaryLabel">Confidence</div>
                          <div className="summaryValue">
                            {result.confidence
                              ? `${(result.confidence * 100).toFixed(1)}%`
                              : "—"}
                          </div>
                          <div className="summaryHint">
                            Probability of fracture outcome.
                          </div>
                        </div>
                        <div className="summaryTile">
                          <div className="summaryLabel">Model</div>
                          <div className="summaryValue">{result.model}</div>
                          <div className="summaryHint">
                            Loaded once from cached registry.
                          </div>
                        </div>
                      </div>
                    ) : null}

                    {result?.warnings?.length ? (
                      <div className="alert alert--warn">
                        {result.warnings.map((warning, index) => (
                          <div key={index}>{warning}</div>
                        ))}
                      </div>
                    ) : null}

                    {result?.valid === false ? (
                      <div className="alert alert--error" role="alert">
                        <strong>
                          {result.validation?.message ?? "Validation failed."}
                        </strong>
                        {result.validation?.reasons?.length ? (
                          <ul>
                            {result.validation.reasons.map((reason, index) => (
                              <li key={index}>{reason}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    ) : null}

                    <div className="explanation">
                      <div className="label">Explanation</div>
                      <p>
                        {result?.explanation ??
                          "Model did not supply an explanation."}
                      </p>
                    </div>

                    <div className="gradcamHeader">
                      <div>
                        <div className="label">Grad-CAM</div>
                        <p className="helper">
                          {result?.gradcam?.status ?? "Not generated."}
                          {result?.gradcam?.message
                            ? ` — ${result.gradcam.message}`
                            : ""}
                        </p>
                      </div>
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={showOverlay}
                          onChange={(event) =>
                            setShowOverlay(event.target.checked)
                          }
                        />
                        <span>Show heatmap overlay</span>
                      </label>
                    </div>

                    <div className="imageGrid">
                      <figure>
                        <figcaption className="label">Original</figcaption>
                        <div className="imgBox">
                          {previewUrl ? (
                            <img src={previewUrl} alt="Uploaded X-ray" />
                          ) : (
                            <div className="empty">Upload to see the preview.</div>
                          )}
                        </div>
                      </figure>
                      <figure>
                        <figcaption className="label">Heatmap</figcaption>
                        <div className="imgBox">
                          {showOverlay && overlayUrl ? (
                            <img src={overlayUrl} alt="Grad-CAM overlay" />
                          ) : heatmapUrl ? (
                            <img src={heatmapUrl} alt="Grad-CAM heatmap" />
                          ) : (
                            <div className="empty">No heatmap produced.</div>
                          )}
                        </div>
                      </figure>
                    </div>
                  </>
                ) : null
              ) : (
                <div className="multiGrid">
                  {hydratedMultiResults.map(
                    ({
                      result: entry,
                      overlayUrl: overlay,
                      heatmapUrl: heatmap,
                    }) => (
                      <article
                        key={`${entry.model}-${entry.prediction}-$${
                          entry.confidence ?? "na"
                        }`}
                        className="multiCard"
                      >
                        <header className="multiCardHeader">
                          <div className="label">Model</div>
                          <h3>{entry.model}</h3>
                        </header>
                        <div className="summaryGrid">
                          <div className="summaryTile">
                            <div className="summaryLabel">Prediction</div>
                            <div className="summaryValue">
                              <span className={`tag tag--${entry.prediction}`}>
                                {entry.prediction}
                              </span>
                            </div>
                            <div className="summaryHint">
                              Based on thesis-ready preprocessing.
                            </div>
                          </div>
                          <div className="summaryTile">
                            <div className="summaryLabel">Confidence</div>
                            <div className="summaryValue">
                              {entry.confidence
                                ? `${(entry.confidence * 100).toFixed(1)}%`
                                : "—"}
                            </div>
                            <div className="summaryHint">
                              Probability of fracture outcome.
                            </div>
                          </div>
                        </div>

                        {entry.warnings?.length ? (
                          <div className="alert alert--warn">
                            {entry.warnings.map((warning, index) => (
                              <div key={index}>{warning}</div>
                            ))}
                          </div>
                        ) : null}

                        <div className="explanation">
                          <div className="label">Explanation</div>
                          <p>
                            {entry.explanation ??
                              "Model did not supply an explanation."}
                          </p>
                        </div>

                        <div className="gradcamHeader">
                          <div>
                            <div className="label">Grad-CAM</div>
                            <p className="helper">
                              {entry.gradcam?.status ?? "Not generated."}
                              {entry.gradcam?.message
                                ? ` — ${entry.gradcam.message}`
                                : ""}
                            </p>
                          </div>
                        </div>

                        <div className="imageGrid">
                          <figure>
                            <figcaption className="label">Original</figcaption>
                            <div className="imgBox">
                              {previewUrl ? (
                                <img src={previewUrl} alt="Uploaded X-ray" />
                              ) : (
                                <div className="empty">
                                  Upload to see the preview.
                                </div>
                              )}
                            </div>
                          </figure>
                          <figure>
                            <figcaption className="label">Heatmap</figcaption>
                            <div className="imgBox">
                              {showOverlay && overlay ? (
                                <img
                                  src={overlay}
                                  alt={`Grad-CAM overlay for ${entry.model}`}
                                />
                              ) : heatmap ? (
                                <img
                                  src={heatmap}
                                  alt={`Grad-CAM heatmap for ${entry.model}`}
                                />
                              ) : (
                                <div className="empty">No heatmap produced.</div>
                              )}
                            </div>
                          </figure>
                        </div>

                        {!entry.valid ? (
                          <div className="alert alert--error" role="alert">
                            <strong>
                              {entry.validation?.message ?? "Validation failed."}
                            </strong>
                            {entry.validation?.reasons?.length ? (
                              <ul>
                                {entry.validation.reasons.map((reason, index) => (
                                  <li key={index}>{reason}</li>
                                ))}
                              </ul>
                            ) : null}
                          </div>
                        ) : null}
                      </article>
                    )
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      <footer className="footer">
        <div className="footerContent">
          <div className="footerTitle">Ready for a live demo?</div>
          <p>
            Start with a smoke test: run{" "}
            <code>scripts/smoke_predict.py --summary</code> to confirm your
            backend and models are loaded, then open the frontend to walk
            through predictions and Grad-CAM overlays.
          </p>
        </div>

        <div className="footerBadgeList">
          <span>TensorFlow 2.20 stack</span>
          <span>FastAPI inference service</span>
          <span>React + Vite UI</span>
          <span>Grad-CAM explainability pipeline</span>
        </div>
      </footer>
    </div>
  );
}
