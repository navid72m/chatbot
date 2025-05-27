// HallucinationReport.jsx
import React from 'react';
import PropTypes from 'prop-types';

const HallucinationReport = ({ analysis }) => {
  if (!analysis || analysis.length === 0) return null;

  return (
    <div className="hallucination-report" style={{ marginTop: '1em', background: '#f8f8f8', padding: '1em', borderRadius: '8px' }}>
      <h4 style={{ marginBottom: '0.5em' }}>🔍 Hallucination Analysis</h4>
      <ul style={{ listStyleType: 'none', paddingLeft: 0 }}>
        {analysis.map((item, idx) => (
          <li
            key={idx}
            style={{
              color: item.is_hallucinated ? '#b00020' : '#2e7d32',
              marginBottom: '0.4em',
              fontSize: '0.95em'
            }}
          >
            {item.is_hallucinated ? '🚫' : '✅'} <strong>{item.claim}</strong> — <em>{item.label}</em>
          </li>
        ))}
      </ul>
    </div>
  );
};

HallucinationReport.propTypes = {
  analysis: PropTypes.arrayOf(
    PropTypes.shape({
      claim: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      is_hallucinated: PropTypes.bool.isRequired,
    })
  ).isRequired,
};

export default HallucinationReport;
