import { useState, useEffect, useMemo } from 'react'
import { Filters } from './Filters'
import { ExperienceCard } from './ExperienceCard'
import './App.css'

function getDisplayUsername(experience) {
  return (
    (experience.username || '').trim() ||
    (experience.user_name || '').trim() ||
    (experience.patient_name || '').trim() ||
    ''
  )
}

function filterExperiences(experiences, selectedPatient, searchQuery) {
  let filtered = experiences

  if (selectedPatient) {
    filtered = filtered.filter(
      (e) => (e.patient || e.patient_id || '').trim() === selectedPatient
    )
  }

  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase()
    filtered = filtered.filter((e) => {
      const desc = (e.description || '').toLowerCase()
      const accession = (e.accession || '').toLowerCase()
      const date = (e.date || '').toLowerCase()
      const modality = (e.modality || '').toLowerCase()
      const patientId = (e.patient || e.patient_id || '').toLowerCase()
      const patientName = getDisplayUsername(e).toLowerCase()
      return (
        desc.includes(q) ||
        accession.includes(q) ||
        date.includes(q) ||
        modality.includes(q) ||
        patientId.includes(q) ||
        patientName.includes(q)
      )
    })
  }

  return filtered
}

function App() {
  const [experiences, setExperiences] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedPatient, setSelectedPatient] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState('all')
  const [generationByExperience, setGenerationByExperience] = useState({})

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetch('/api/experiences')
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText || `HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        if (!cancelled && Array.isArray(data)) setExperiences(data)
        else if (!cancelled) setError('Invalid response')
      })
      .catch((err) => {
        if (!cancelled) setError('Failed to load experiences: ' + (err.message || String(err)))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const patients = useMemo(() => {
    const byId = new Map()
    for (const e of experiences) {
      const id = (e.patient || e.patient_id || '').trim()
      if (!id) continue
      const name = getDisplayUsername(e)
      if (!byId.has(id)) byId.set(id, { id, name: name || id })
      else if (name && byId.get(id).name === id) byId.set(id, { id, name })
    }
    return [...byId.values()].sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id))
  }, [experiences])

  const filteredExperiences = useMemo(
    () => filterExperiences(experiences, selectedPatient, searchQuery),
    [experiences, selectedPatient, searchQuery]
  )
  const unvalidatedExperiences = useMemo(
    () =>
      filteredExperiences.filter((exp) =>
        (exp.reports || []).some((report) => !report.is_validated)
      ),
    [filteredExperiences]
  )
  const displayedExperiences = activeTab === 'unvalidated' ? unvalidatedExperiences : filteredExperiences

  const handleGenerateReport = async (experience) => {
    const experienceKey = experience.id
    const patientId = (experience.patient || experience.patient_id || '').trim()
    if (!patientId) {
      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          loading: false,
          error: 'Missing patient identifier.',
          report: '',
          saveError: '',
          saveSuccess: false,
          saveLoading: false,
        },
      }))
      return
    }

    setGenerationByExperience((prev) => ({
      ...prev,
      [experienceKey]: {
        ...prev[experienceKey],
        loading: true,
        error: '',
        report: '',
        saveError: '',
        saveSuccess: false,
      },
    }))

    try {
      const res = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: patientId }),
      })
      const data = await res.json()
      if (!res.ok) {
        const detail = data?.detail || res.statusText || `HTTP ${res.status}`
        throw new Error(detail)
      }

      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          ...prev[experienceKey],
          loading: false,
          error: '',
          report: data.report || '',
          saveError: '',
          saveSuccess: false,
        },
      }))
    } catch (err) {
      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          ...prev[experienceKey],
          loading: false,
          error: 'Failed to generate report: ' + (err.message || String(err)),
          report: '',
          saveError: '',
          saveSuccess: false,
        },
      }))
    }
  }

  const handleSaveGeneratedReport = async (experience, reportText) => {
    const experienceKey = experience.id
    const patientId = (experience.patient || experience.patient_id || '').trim()
    const experienceId = (experience.accession || '').trim()
    const reportDescription = (reportText || '').trim()

    if (!patientId || !experienceId || !reportDescription) {
      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          ...prev[experienceKey],
          saveLoading: false,
          saveSuccess: false,
          saveError: 'Missing patient, experience or report content.',
        },
      }))
      return
    }

    setGenerationByExperience((prev) => ({
      ...prev,
      [experienceKey]: {
        ...prev[experienceKey],
        saveLoading: true,
        saveSuccess: false,
        saveError: '',
      },
    }))

    try {
      const res = await fetch('/api/reports/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          experience_id: experienceId,
          report_description: reportDescription,
          report_type: 'Generated',
          is_validated: true,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        const detail = data?.detail || res.statusText || `HTTP ${res.status}`
        throw new Error(detail)
      }

      const savedReport = data?.report
      if (savedReport) {
        setExperiences((prev) =>
          prev.map((exp) => {
            if (exp.id !== experienceKey) return exp
            const reports = exp.reports || []
            const existingIdx = reports.findIndex((r) => String(r.id) === String(savedReport.id))
            if (existingIdx >= 0) {
              const updated = [...reports]
              updated[existingIdx] = savedReport
              return { ...exp, reports: updated }
            }
            return { ...exp, reports: [savedReport, ...reports] }
          })
        )
      }

      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          ...prev[experienceKey],
          saveLoading: false,
          saveSuccess: true,
          saveError: '',
          report: reportDescription,
        },
      }))
    } catch (err) {
      setGenerationByExperience((prev) => ({
        ...prev,
        [experienceKey]: {
          ...prev[experienceKey],
          saveLoading: false,
          saveSuccess: false,
          saveError: 'Failed to save report: ' + (err.message || String(err)),
        },
      }))
    }
  }

  if (loading) {
    return (
      <div className="container">
        <h1>Experiences</h1>
        <p className="subtitle">Orthanc studies and linked reports from the database.</p>
        <div className="loading">Loading experiences…</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container">
        <h1>Experiences</h1>
        <p className="subtitle">Orthanc studies and linked reports from the database.</p>
        <div className="error">{error}</div>
      </div>
    )
  }

  return (
    <div className="container">
      <h1>Experiences</h1>
      <p className="subtitle">Orthanc studies and linked reports from the database.</p>
      <Filters
        patients={patients}
        selectedPatient={selectedPatient}
        onPatientChange={setSelectedPatient}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />
      <div className="tabs" role="tablist" aria-label="Experience filters">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'all'}
          className={`tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          All experiences ({filteredExperiences.length})
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'unvalidated'}
          className={`tab ${activeTab === 'unvalidated' ? 'active' : ''}`}
          onClick={() => setActiveTab('unvalidated')}
        >
          Unvalidated reports ({unvalidatedExperiences.length})
        </button>
      </div>
      {displayedExperiences.map((exp) => (
        <ExperienceCard
          key={exp.id}
          experience={exp}
          onGenerateReport={handleGenerateReport}
          onSaveGeneratedReport={handleSaveGeneratedReport}
          generationState={generationByExperience[exp.id]}
        />
      ))}
      {displayedExperiences.length === 0 && (
        <div className="loading">No experiences match the current filters.</div>
      )}
    </div>
  )
}

export default App
