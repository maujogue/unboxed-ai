/**
 * Finding images are loaded at runtime from the backend, not from Vite.
 *
 * Flow:
 * 1. Generate report → backend runs generate_nodule_images.py → PNGs written to
 *    export_nodules/{patient_id}/{accession}_findingN_...png on the server.
 * 2. Backend returns nodule_images: [{ url: "/api/nodule-images/static/...", filename }].
 * 3. We render <img src={url} />. The browser requests that URL.
 * 4. In dev (Vite): the page is on e.g. localhost:5173; vite.config.js proxies /api → localhost:8000,
 *    so the image request goes to the FastAPI server, which serves the file from export_nodules/.
 * 5. Backend must be running (e.g. uvicorn on port 8000) for /api and images to work.
 *
 * If images don't appear: ensure the backend is running, then check DevTools → Network
 * for /api/nodule-images (list) and /api/nodule-images/static/... (each image); 404 = path or server issue.
 */
import { useEffect, useState } from 'react'

const FALLBACK_IMAGES = [
  { url: 'https://placehold.co/200x200/1a1a2e/eee?text=F1', filename: 'Finding 1' },
  { url: 'https://placehold.co/200x200/16213e/eee?text=F2', filename: 'Finding 2' },
  { url: 'https://placehold.co/200x200/0f3460/eee?text=F3', filename: 'Finding 3' },
]

export function NoduleImagesList({ patientId, accessionId, initialImages }) {
  const [images, setImages] = useState(
    () => (Array.isArray(initialImages) && initialImages.length > 0 ? initialImages : null)
  )
  const [loading, setLoading] = useState(
    () =>
      !(Array.isArray(initialImages) && initialImages.length > 0) &&
      !!(patientId?.trim() && accessionId?.trim())
  )

  useEffect(() => {
    if (Array.isArray(initialImages) && initialImages.length > 0) {
      setImages(initialImages)
      setLoading(false)
      return
    }
    if (!patientId?.trim() || !accessionId?.trim()) {
      setImages([])
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    const params = new URLSearchParams({
      patient_id: patientId.trim(),
      accession_id: accessionId.trim(),
    })
    fetch(`/api/nodule-images?${params}`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setImages(Array.isArray(data?.images) ? data.images : [])
      })
      .catch(() => {
        if (!cancelled) setImages([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [patientId, accessionId, initialImages])

  const list = images?.length ? images : FALLBACK_IMAGES
  if (loading) {
    return (
      <div className="nodule-images-list nodule-images-list--loading">
        <span>Loading finding images…</span>
      </div>
    )
  }

  return (
    <div className="nodule-images-list">
      <div className="nodule-images-list-title">Finding images</div>
      <div className="nodule-images-list-grid">
        {list.map((img, i) => (
          <figure key={i} className="nodule-images-list-item">
            <img
              src={img.url}
              alt={img.filename || `Finding ${i + 1}`}
              className="nodule-images-list-img"
            />
            <figcaption className="nodule-images-list-caption">{img.filename}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  )
}
