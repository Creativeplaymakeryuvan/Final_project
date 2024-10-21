import React from 'react';
import './sortBy.css'

const SortBy = ({
  sortBy,
  handleSortByChange,
  handleMonthChange,
  handleYearChange,
  currentMonth,
  currentYear,
  month,
  year,
}) => {
  return (
    <div>

      {sortBy === 'custom' && (
        <div className="date-selection">
          <label htmlFor="month">Month:</label>
          <select id="month" value={month || currentMonth} onChange={(e) => handleMonthChange(Number(e.target.value))}>
            {[...Array(12).keys()].map((m) => (
              <option key={m + 1} value={m + 1}>
                {new Date(0, m).toLocaleString('default', { month: 'long' })}
              </option>
            ))}
          </select>

          <label htmlFor="year">Year:</label>
          <input
            id="year"
            type="number"
            value={year || currentYear}
            onChange={(e) => handleYearChange(Number(e.target.value))}
            min="2000"
            max={currentYear}
          />
        </div>
      )}
    </div>
  );
};

export default SortBy;
