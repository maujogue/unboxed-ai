import { useState } from 'react'
import { MarkdownContent } from './MarkdownContent'

const TRUNCATE_LEN = 320

export function ReportCard({ report }) {
  const desc = report.report_description || ''
  const short = desc.length > TRUNCATE_LEN
  const [expanded, setExpanded] = useState(false)
  const validatedClass = report.is_validated ? 'yes' : 'no'
  const validatedLabel = report.is_validated ? 'Validated' : 'Not validated'

  return (
    <div className="report" data-report-id={report.id}>
      <div className="report-header">
        <span className="report-type">{report.type}</span>
        <span className={`report-validated ${validatedClass}`}>{validatedLabel}</span>
      </div>
      <div className={`report-description ${expanded ? 'expanded' : ''}`}>
        <MarkdownContent content={desc} />
      </div>
      {short && (
        <button
          type="button"
          className="toggle-desc"
          onClick={() => setExpanded((e) => !e)}
          aria-label={expanded ? 'Collapse description' : 'Expand description'}
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}
