// src/pages/users/UserList.jsx

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  FaUserPlus, 
  FaEye, 
  FaEdit, 
  FaTrash, 
  FaDownload, 
  FaFilter, 
  FaUserShield
} from 'react-icons/fa';

// Components
import Button from '../../components/common/Button';
import DataTable from '../../components/data/DataTable';
import Modal from '../../components/common/Modal';
import Card from '../../components/common/Card';
import Input from '../../components/common/Input';
import { useNotification } from '../../context/NotificationContext';

// API
import api from '../../api/client';

const UserList = () => {
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // State for users data
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState([]);
  const [totalUsers, setTotalUsers] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  
  // State for sorting and filtering
  const [sortField, setSortField] = useState('createdAt');
  const [sortDirection, setSortDirection] = useState('desc');
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    role: '',
    status: '',
  });
  
  // State for delete confirmation
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  
  // Fetch users data
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        setLoading(true);
        
        // Build query parameters
        const params = {
          page: currentPage,
          limit: pageSize,
          sort: sortField,
          order: sortDirection,
          search: searchTerm,
          ...filters,
        };
        
        // This would be your actual API call
        // const response = await api.get('/users', { params });
        
        // Simulated API response for demonstration
        const response = {
          data: [
            { id: 1, name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active', createdAt: '2023-01-15' },
            { id: 2, name: 'Jane Smith', email: 'jane@example.com', role: 'Editor', status: 'Active', createdAt: '2023-02-20' },
            { id: 3, name: 'Bob Johnson', email: 'bob@example.com', role: 'User', status: 'Inactive', createdAt: '2023-03-10' },
            { id: 4, name: 'Alice Brown', email: 'alice@example.com', role: 'User', status: 'Active', createdAt: '2023-04-05' },
            { id: 5, name: 'David Wilson', email: 'david@example.com', role: 'Editor', status: 'Active', createdAt: '2023-05-12' },
            { id: 6, name: 'Sarah Miller', email: 'sarah@example.com', role: 'User', status: 'Active', createdAt: '2023-06-18' },
            { id: 7, name: 'Michael Brown', email: 'michael@example.com', role: 'User', status: 'Inactive', createdAt: '2023-07-22' },
            { id: 8, name: 'Emma Davis', email: 'emma@example.com', role: 'Editor', status: 'Active', createdAt: '2023-08-14' },
            { id: 9, name: 'James Wilson', email: 'james@example.com', role: 'User', status: 'Active', createdAt: '2023-09-03' },
            { id: 10, name: 'Olivia Taylor', email: 'olivia@example.com', role: 'Admin', status: 'Active', createdAt: '2023-10-29' },
          ],
          total: 42,
        };
        
        setUsers(response.data);
        setTotalUsers(response.total);
      } catch (error) {
        console.error('Error fetching users:', error);
        showError('Failed to load users');
      } finally {
        setLoading(false);
      }
    };
    
    fetchUsers();
  }, [currentPage, pageSize, sortField, sortDirection, searchTerm, filters, showError]);
  
  // Handle page change
  const handlePageChange = (page) => {
    setCurrentPage(page);
  };
  
  // Handle sort
  const handleSort = (field, direction) => {
    setSortField(field);
    setSortDirection(direction);
  };
  
  // Handle search
  const handleSearch = (value) => {
    setSearchTerm(value);
    setCurrentPage(1); // Reset to first page
  };
  
  // Handle filter change
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setCurrentPage(1); // Reset to first page
  };
  
  // Handle user actions
  const handleViewUser = (user) => {
    navigate(`/users/${user.id}`);
  };
  
  const handleEditUser = (user) => {
    navigate(`/users/${user.id}/edit`);
  };
  
  const handleDeleteClick = (user) => {
    setUserToDelete(user);
    setDeleteModalOpen(true);
  };
  
  const handleDeleteUser = async () => {
    if (!userToDelete) return;
    
    try {
      // This would be your actual API call
      // await api.delete(`/users/${userToDelete.id}`);
      
      // Simulated API call for demonstration
      console.log(`Deleting user: ${userToDelete.id}`);
      
      // Update local state
      setUsers(users.filter(user => user.id !== userToDelete.id));
      
      showSuccess(`User "${userToDelete.name}" has been deleted`);
    } catch (error) {
      console.error('Error deleting user:', error);
      showError('Failed to delete user');
    } finally {
      setDeleteModalOpen(false);
      setUserToDelete(null);
    }
  };
  
  const handleCreateUser = () => {
    navigate('/users/create');
  };
  
  // Table columns configuration
  const columns = [
    { field: 'id', header: 'ID', width: '70px' },
    { field: 'name', header: 'Name' },
    { field: 'email', header: 'Email' },
    { 
      field: 'role', 
      header: 'Role',
      render: (value) => {
        let roleClass = '';
        
        switch (value) {
          case 'Admin':
            roleClass = 'primary';
            break;
          case 'Editor':
            roleClass = 'info';
            break;
          default:
            roleClass = 'secondary';
        }
        
        return <span className={`badge bg-${roleClass}`}>{value}</span>;
      }
    },
    { 
      field: 'status', 
      header: 'Status',
      render: (value) => {
        const statusClass = value === 'Active' ? 'success' : 'danger';
        return <span className={`badge bg-${statusClass}`}>{value}</span>;
      }
    },
    { 
      field: 'createdAt', 
      header: 'Created At',
      render: (value) => new Date(value).toLocaleDateString(),
    },
  ];
  
  // Table row actions
  const rowActions = [
    {
      icon: <FaEye />,
      title: 'View User',
      buttonClass: 'btn-outline-primary',
      onClick: handleViewUser,
    },
    {
      icon: <FaEdit />,
      title: 'Edit User',
      buttonClass: 'btn-outline-secondary',
      onClick: handleEditUser,
    },
    {
      icon: <FaTrash />,
      title: 'Delete User',
      buttonClass: 'btn-outline-danger',
      onClick: handleDeleteClick,
    },
  ];
  
  // Table actions
  const tableActions = [
    {
      icon: <FaUserPlus />,
      label: 'Add User',
      buttonClass: 'btn-primary',
      onClick: handleCreateUser,
    },
    {
      icon: <FaDownload />,
      label: 'Export',
      buttonClass: 'btn-outline-secondary',
      onClick: () => console.log('Export users'),
    },
  ];
  
  // Table filters
  const tableFilters = [
    {
      id: 'role',
      label: 'Role',
      type: 'select',
      options: [
        { value: '', label: 'All Roles' },
        { value: 'Admin', label: 'Admin' },
        { value: 'Editor', label: 'Editor' },
        { value: 'User', label: 'User' },
      ],
    },
    {
      id: 'status',
      label: 'Status',
      type: 'select',
      options: [
        { value: '', label: 'All Statuses' },
        { value: 'Active', label: 'Active' },
        { value: 'Inactive', label: 'Inactive' },
      ],
    },
  ];
  
  return (
    <div className="user-list-page">
      {/* Page header */}
      <div className="page-header mb-4">
        <h1 className="page-title">User Management</h1>
        <div className="page-actions">
          <Button 
            variant="primary" 
            icon={<FaUserPlus />}
            onClick={handleCreateUser}
          >
            Add User
          </Button>
        </div>
      </div>
      
      {/* Users overview */}
      <div className="row mb-4">
        <div className="col-md-4 mb-4 mb-md-0">
          <Card
            className="dashboard-stat-card"
            icon={<FaUserShield />}
            title={loading ? 'Loading...' : totalUsers.toString()}
            subtitle="Total Users"
          />
        </div>
        
        <div className="col-md-8">
          <Card>
            <div className="user-list-filters">
              <h5>Quick Filters</h5>
              <div className="filter-buttons">
                <Button 
                  size="sm" 
                  variant={filters.role === 'Admin' ? 'primary' : 'outline-primary'}
                  onClick={() => handleFilterChange({ ...filters, role: filters.role === 'Admin' ? '' : 'Admin' })}
                >
                  Admins
                </Button>
                <Button 
                  size="sm" 
                  variant={filters.status === 'Inactive' ? 'primary' : 'outline-primary'}
                  onClick={() => handleFilterChange({ ...filters, status: filters.status === 'Inactive' ? '' : 'Inactive' })}
                >
                  Inactive Users
                </Button>
                <Button 
                  size="sm" 
                  variant={filters.role === 'Editor' ? 'primary' : 'outline-primary'}
                  onClick={() => handleFilterChange({ ...filters, role: filters.role === 'Editor' ? '' : 'Editor' })}
                >
                  Editors
                </Button>
                <Button 
                  size="sm" 
                  variant="outline-secondary"
                  onClick={() => handleFilterChange({ role: '', status: '' })}
                >
                  Clear All
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
      
      {/* Users table */}
      <DataTable
        title="Users"
        data={users}
        columns={columns}
        loading={loading}
        pagination={true}
        pageSize={pageSize}
        currentPage={currentPage}
        totalItems={totalUsers}
        onPageChange={handlePageChange}
        sortable={true}
        defaultSortField="createdAt"
        defaultSortDirection="desc"
        onSort={handleSort}
        searchable={true}
        onSearch={handleSearch}
        filters={tableFilters}
        onFilter={handleFilterChange}
        rowActions={rowActions}
        tableActions={tableActions}
        onRowClick={handleViewUser}
        emptyStateMessage="No users found matching your criteria"
      />
      
      {/* Delete confirmation modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Confirm Delete"
        footer={
          <>
            <Button 
              variant="outline-secondary" 
              onClick={() => setDeleteModalOpen(false)}
            >
              Cancel
            </Button>
            <Button 
              variant="danger" 
              onClick={handleDeleteUser}
            >
              Delete
            </Button>
          </>
        }
      >
        {userToDelete && (
          <p>
            Are you sure you want to delete the user "{userToDelete.name}"? This action cannot be undone.
          </p>
        )}
      </Modal>
    </div>
  );
};

export default UserList;