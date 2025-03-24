// src/components/data/DataTable.jsx

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import { FaSort, FaSortUp, FaSortDown, FaSearch, FaFilter, FaPlus, FaEdit, FaTrash, FaDownload, FaUpload } from 'react-icons/fa';

const DataTable = ({
  data = [],
  columns = [],
  title = '',
  loading = false,
  pagination = true,
  pageSize = 10,
  currentPage: controlledCurrentPage,
  totalItems: controlledTotalItems,
  onPageChange = null,
  sortable = true,
  defaultSortField = '',
  defaultSortDirection = 'asc',
  onSort = null,
  searchable = true,
  onSearch = null,
  filters = [],
  onFilter = null,
  selectable = false,
  onRowSelect = null,
  onRowClick = null,
  rowActions = [],
  tableActions = [],
  emptyStateMessage = 'No data found',
  className = '',
}) => {
  // State for internal pagination if not controlled from parent
  const [internalCurrentPage, setInternalCurrentPage] = useState(1);
  const [internalPageSize] = useState(pageSize);
  
  // Use controlled pagination if provided, otherwise use internal
  const currentPage = controlledCurrentPage || internalCurrentPage;
  
  // State for sorting
  const [sortField, setSortField] = useState(defaultSortField);
  const [sortDirection, setSortDirection] = useState(defaultSortDirection);
  
  // State for searching
  const [searchTerm, setSearchTerm] = useState('');
  
  // State for filtering
  const [activeFilters, setActiveFilters] = useState({});
  
  // State for selected rows
  const [selectedRows, setSelectedRows] = useState([]);
  
  // Handle page change
  const handlePageChange = useCallback(
    (page) => {
      if (onPageChange) {
        onPageChange(page);
      } else {
        setInternalCurrentPage(page);
      }
    },
    [onPageChange]
  );
  
  // Handle sorting
  const handleSort = useCallback(
    (field) => {
      if (!sortable) return;
      
      const newDirection =
        field === sortField
          ? sortDirection === 'asc'
            ? 'desc'
            : 'asc'
          : 'asc';
      
      setSortField(field);
      setSortDirection(newDirection);
      
      if (onSort) {
        onSort(field, newDirection);
      }
    },
    [sortable, sortField, sortDirection, onSort]
  );
  
  // Handle search
  const handleSearch = useCallback(
    (e) => {
      const { value } = e.target;
      setSearchTerm(value);
      
      if (onSearch) {
        onSearch(value);
      }
    },
    [onSearch]
  );
  
  // Handle filter change
  const handleFilterChange = useCallback(
    (filterId, value) => {
      setActiveFilters((prev) => ({
        ...prev,
        [filterId]: value,
      }));
      
      if (onFilter) {
        onFilter({ ...activeFilters, [filterId]: value });
      }
    },
    [activeFilters, onFilter]
  );
  
  // Handle row selection
  const handleRowSelect = useCallback(
    (rowId, isSelected) => {
      if (!selectable) return;
      
      setSelectedRows((prev) => {
        if (isSelected) {
          return [...prev, rowId];
        } else {
          return prev.filter((id) => id !== rowId);
        }
      });
      
      if (onRowSelect) {
        onRowSelect(rowId, isSelected);
      }
    },
    [selectable, onRowSelect]
  );
  
  // Handle select all rows
  const handleSelectAll = useCallback(
    (isSelected) => {
      if (!selectable) return;
      
      if (isSelected) {
        const allIds = data.map((row) => row.id);
        setSelectedRows(allIds);
        
        if (onRowSelect) {
          onRowSelect(allIds, true);
        }
      } else {
        setSelectedRows([]);
        
        if (onRowSelect) {
          onRowSelect([], false);
        }
      }
    },
    [selectable, data, onRowSelect]
  );
  
  // Calculate paginated data for internal pagination
  const paginatedData = useMemo(() => {
    if (controlledTotalItems !== undefined) {
      // If pagination is controlled by parent, return full data
      return data;
    }
    
    // Otherwise, paginate internally
    const startIndex = (internalCurrentPage - 1) * internalPageSize;
    const endIndex = startIndex + internalPageSize;
    return data.slice(startIndex, endIndex);
  }, [data, controlledTotalItems, internalCurrentPage, internalPageSize]);
  
  // Calculate total pages for internal pagination
  const totalPages = useMemo(() => {
    if (controlledTotalItems !== undefined) {
      return Math.ceil(controlledTotalItems / pageSize);
    }
    
    return Math.ceil(data.length / internalPageSize);
  }, [data.length, controlledTotalItems, internalPageSize, pageSize]);
  
  // Calculate total items for internal pagination
  const totalItems = useMemo(() => {
    return controlledTotalItems !== undefined ? controlledTotalItems : data.length;
  }, [data.length, controlledTotalItems]);
  
  // Check if all rows are selected
  const allRowsSelected = useMemo(() => {
    return data.length > 0 && selectedRows.length === data.length;
  }, [data.length, selectedRows.length]);
  
  // Reset pagination when data changes
  useEffect(() => {
    if (!onPageChange) {
      setInternalCurrentPage(1);
    }
  }, [data.length, onPageChange]);
  
  // Render pagination controls
  const renderPagination = () => {
    if (!pagination || totalPages <= 1) {
      return null;
    }
    
    const pages = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    // Add first page
    if (startPage > 1) {
      pages.push(
        <li
          key="first"
          className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}
        >
          <button
            className="page-link"
            onClick={() => handlePageChange(1)}
            disabled={currentPage === 1}
          >
            First
          </button>
        </li>
      );
    }
    
    // Add previous page
    pages.push(
      <li
        key="prev"
        className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}
      >
        <button
          className="page-link"
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
      </li>
    );
    
    // Add page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <li
          key={i}
          className={`page-item ${currentPage === i ? 'active' : ''}`}
        >
          <button className="page-link" onClick={() => handlePageChange(i)}>
            {i}
          </button>
        </li>
      );
    }
    
    // Add next page
    pages.push(
      <li
        key="next"
        className={`page-item ${currentPage === totalPages ? 'disabled' : ''}`}
      >
        <button
          className="page-link"
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </li>
    );
    
    // Add last page
    if (endPage < totalPages) {
      pages.push(
        <li
          key="last"
          className={`page-item ${currentPage === totalPages ? 'disabled' : ''}`}
        >
          <button
            className="page-link"
            onClick={() => handlePageChange(totalPages)}
            disabled={currentPage === totalPages}
          >
            Last
          </button>
        </li>
      );
    }
    
    return (
      <div className="data-table-pagination">
        <div className="pagination-info">
          Showing {Math.min((currentPage - 1) * pageSize + 1, totalItems)} to{' '}
          {Math.min(currentPage * pageSize, totalItems)} of {totalItems} entries
        </div>
        <nav>
          <ul className="pagination">{pages}</ul>
        </nav>
      </div>
    );
  };
  
  // Render sorting icon
  const renderSortIcon = (field) => {
    if (!sortable) return null;
    
    if (field !== sortField) {
      return <FaSort className="sort-icon" />;
    }
    
    return sortDirection === 'asc' ? (
      <FaSortUp className="sort-icon active" />
    ) : (
      <FaSortDown className="sort-icon active" />
    );
  };
  
  // Render table header
  const renderTableHeader = () => {
    return (
      <thead>
        <tr>
          {selectable && (
            <th className="select-column">
              <input
                type="checkbox"
                checked={allRowsSelected}
                onChange={(e) => handleSelectAll(e.target.checked)}
                disabled={data.length === 0}
              />
            </th>
          )}
          
          {columns.map((column) => (
            <th
              key={column.field}
              className={column.sortable === false ? '' : 'sortable'}
              onClick={() =>
                column.sortable !== false && handleSort(column.field)
              }
              style={{ width: column.width || 'auto' }}
            >
              {column.header}
              {column.sortable !== false && renderSortIcon(column.field)}
            </th>
          ))}
          
          {rowActions.length > 0 && <th className="actions-column">Actions</th>}
        </tr>
      </thead>
    );
  };
  
  // Render table body
  const renderTableBody = () => {
    if (loading) {
      return (
        <tbody>
          <tr>
            <td
              colSpan={
                columns.length + (selectable ? 1 : 0) + (rowActions.length > 0 ? 1 : 0)
              }
              className="text-center"
            >
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
            </td>
          </tr>
        </tbody>
      );
    }
    
    if (paginatedData.length === 0) {
      return (
        <tbody>
          <tr>
            <td
              colSpan={
                columns.length + (selectable ? 1 : 0) + (rowActions.length > 0 ? 1 : 0)
              }
              className="text-center"
            >
              {emptyStateMessage}
            </td>
          </tr>
        </tbody>
      );
    }
    
    return (
      <tbody>
        {paginatedData.map((row) => (
          <tr
            key={row.id}
            className={selectedRows.includes(row.id) ? 'selected' : ''}
            onClick={() => onRowClick && onRowClick(row)}
            style={{ cursor: onRowClick ? 'pointer' : 'default' }}
          >
            {selectable && (
              <td
                className="select-column"
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={selectedRows.includes(row.id)}
                  onChange={(e) =>
                    handleRowSelect(row.id, e.target.checked)
                  }
                />
              </td>
            )}
            
            {columns.map((column) => (
              <td key={`${row.id}-${column.field}`}>
                {column.render
                  ? column.render(row[column.field], row)
                  : row[column.field]}
              </td>
            ))}
            
            {rowActions.length > 0 && (
              <td className="actions-column" onClick={(e) => e.stopPropagation()}>
                <div className="d-flex">
                  {rowActions.map((action, index) => (
                    <button
                      key={index}
                      className={`btn btn-sm ${action.buttonClass || 'btn-outline-primary'} me-1`}
                      onClick={() => action.onClick(row)}
                      disabled={action.isDisabled ? action.isDisabled(row) : false}
                      title={action.title}
                    >
                      {action.icon}
                      {action.label && <span className="ms-1">{action.label}</span>}
                    </button>
                  ))}
                </div>
              </td>
            )}
          </tr>
        ))}
      </tbody>
    );
  };
  
  // Render filters
  const renderFilters = () => {
    if (!filters || filters.length === 0) {
      return null;
    }
    
    return (
      <div className="data-table-filters">
        {filters.map((filter) => (
          <div key={filter.id} className="filter-item">
            <label>{filter.label}</label>
            {filter.type === 'select' ? (
              <select
                className="form-select"
                value={activeFilters[filter.id] || ''}
                onChange={(e) => handleFilterChange(filter.id, e.target.value)}
              >
                <option value="">All</option>
                {filter.options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : filter.type === 'date' ? (
              <input
                type="date"
                className="form-control"
                value={activeFilters[filter.id] || ''}
                onChange={(e) => handleFilterChange(filter.id, e.target.value)}
              />
            ) : filter.type === 'boolean' ? (
              <select
                className="form-select"
                value={activeFilters[filter.id] || ''}
                onChange={(e) => handleFilterChange(filter.id, e.target.value)}
              >
                <option value="">All</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            ) : (
              <input
                type="text"
                className="form-control"
                value={activeFilters[filter.id] || ''}
                onChange={(e) => handleFilterChange(filter.id, e.target.value)}
                placeholder={filter.placeholder || `Filter by ${filter.label}`}
              />
            )}
          </div>
        ))}
        
        <button
          className="btn btn-outline-secondary"
          onClick={() => {
            setActiveFilters({});
            if (onFilter) {
              onFilter({});
            }
          }}
        >
          Clear Filters
        </button>
      </div>
    );
  };

  return (
    <div className={`data-table-container ${className}`}>
      {title && <h2 className="data-table-title">{title}</h2>}
      
      {searchable && (
        <div className="data-table-search">
          <FaSearch className="search-icon" />
          <input
            type="text"
            className="form-control"
            placeholder="Search..."
            value={searchTerm}
            onChange={handleSearch}
          />
        </div>
      )}
      
      {renderFilters()}
      
      <div className="table-responsive">
        <table className="table">
          {renderTableHeader()}
          {renderTableBody()}
        </table>
      </div>
      
      {renderPagination()}
      
      {tableActions.length > 0 && (
        <div className="data-table-actions">
          {tableActions.map((action, index) => (
            <button
              key={index}
              className={`btn ${action.buttonClass || 'btn-primary'}`}
              onClick={action.onClick}
            >
              {action.icon}
              {action.label && <span className="ms-1">{action.label}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

DataTable.propTypes = {
  data: PropTypes.array,
  columns: PropTypes.array,
  title: PropTypes.string,
  loading: PropTypes.bool,
  pagination: PropTypes.bool,
  pageSize: PropTypes.number,
  currentPage: PropTypes.number,
  totalItems: PropTypes.number,
  onPageChange: PropTypes.func,
  sortable: PropTypes.bool,
  defaultSortField: PropTypes.string,
  defaultSortDirection: PropTypes.string,
  onSort: PropTypes.func,
  searchable: PropTypes.bool,
  onSearch: PropTypes.func,
  filters: PropTypes.array,
  onFilter: PropTypes.func,
  selectable: PropTypes.bool,
  onRowSelect: PropTypes.func,
  onRowClick: PropTypes.func,
  rowActions: PropTypes.array,
  tableActions: PropTypes.array,
  emptyStateMessage: PropTypes.string,
  className: PropTypes.string
};

export default DataTable;