export function Filters({ patients, selectedPatient, onPatientChange, searchQuery, onSearchChange }) {
  return (
    <div className="filters">
      <label>
        Patient:
        <select
          value={selectedPatient}
          onChange={(e) => onPatientChange(e.target.value)}
          aria-label="Filter by patient"
        >
          <option value="">All</option>
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name || p.id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Search experiences:
        <input
          type="text"
          placeholder="Description, accession, date..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          aria-label="Search experiences"
        />
      </label>
    </div>
  )
}
