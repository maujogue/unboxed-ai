import { useEffect, useState } from 'react'
import { ReportCard } from './ReportCard'
import { MarkdownContent } from './MarkdownContent'
import { NoduleImagesList } from './NoduleImagesList'

function formatDate(d) {
  if (!d || d === '-') return d
  const s = String(d)
  if (s.length === 8) return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`
  return s
}

function displayValue(val, fallback = '—') {
  if (!val || val === '-') return fallback
  return val
}

function getDisplayUsername(experience) {
  return (
    (experience.username || '').trim() ||
    (experience.user_name || '').trim() ||
    (experience.patient_name || '').trim() ||
    ''
  )
}

export function ExperienceCard({
  experience,
  onGenerateReport,
  onSaveGeneratedReport,
  generationState,
}) {
  const [expanded, setExpanded] = useState(false)
  const [isEditingGenerated, setIsEditingGenerated] = useState(false)
  const [draftReport, setDraftReport] = useState('')
  const reports = experience.reports || []
  const reportsCount = reports.length
  const reportsLabel =
    reportsCount === 0 ? 'No reports' : `${reportsCount} report${reportsCount !== 1 ? 's' : ''}`

  useEffect(() => {
    setDraftReport(generationState?.report || '')
    setIsEditingGenerated(false)
  }, [generationState?.report, experience.id])

  return (
    <div className="experience">
      <div
        className="experience-header"
        onClick={() => setExpanded((e) => !e)}
        role="button"
        tabIndex={0}
        onKeyDown={(ev) => ev.key === 'Enter' && setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <div>
          <div className="experience-id">{experience.id}</div>
          <div>{experience.description || '—'}</div>
        </div>
        <div className="experience-meta">
          <span>
            Patient{' '}
            {displayValue(getDisplayUsername(experience) || experience.patient || experience.patient_id)}
          </span>
          <span>Accession {displayValue(experience.accession)}</span>
          <span>{formatDate(experience.date)}</span>
          <span>{experience.modality || '—'}</span>
        </div>
        <span className={`reports-count ${reportsCount === 0 ? 'none' : ''}`}>{reportsLabel}</span>
      </div>
      {expanded && (
        <div className="experience-body">
          <div className="experience-actions">
            <button
              type="button"
              className="generate-report-btn"
              onClick={() => onGenerateReport(experience)}
              disabled={generationState?.loading}
            >
              {generationState?.loading ? 'Generating…' : 'Generate report'}
            </button>
          </div>
          {generationState?.error && <div className="generation-error">{generationState.error}</div>}
          {generationState?.report && (
            <div className="generated-report-block">
              <div className="generated-report">
                <div className="generated-report-title">Generated report</div>
                <div className="generated-report-toolbar">
                  {!isEditingGenerated ? (
                    <button
                      type="button"
                      className="toggle-desc"
                      onClick={() => setIsEditingGenerated(true)}
                    >
                      Edit markdown
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="toggle-desc"
                      onClick={() => {
                        setDraftReport(generationState.report || '')
                        setIsEditingGenerated(false)
                      }}
                    >
                      Cancel edits
                    </button>
                  )}
                </div>
                {isEditingGenerated ? (
                  <textarea
                    className="generated-report-editor"
                    value={draftReport}
                    onChange={(e) => setDraftReport(e.target.value)}
                    rows={10}
                    aria-label="Edit generated markdown report"
                  />
                ) : (
                  <div className="generated-report-content">
                    <MarkdownContent content={draftReport || generationState.report} />
                  </div>
                )}
                <div className="generated-report-toolbar">
                  <button
                    type="button"
                    className="generate-report-btn"
                    onClick={() => onSaveGeneratedReport(experience, draftReport)}
                    disabled={generationState?.saveLoading || !draftReport.trim()}
                  >
                    {generationState?.saveLoading ? 'Validating…' : 'Validate and save report'}
                  </button>
                </div>
                {generationState?.saveError && (
                  <div className="generation-error">{generationState.saveError}</div>
                )}
                {generationState?.saveSuccess && (
                  <div className="generation-success">Report saved and validated.</div>
                )}
              </div>
              <NoduleImagesList
                patientId={experience.patient ?? experience.patient_id}
                accessionId={experience.accession ?? experience.accession_id}
                initialImages={generationState.noduleImages}
              />
            </div>
          )}
          {reports.map((r) => (
            <ReportCard key={r.id} report={r} />
          ))}
        </div>
      )}
    </div>
  )
}
