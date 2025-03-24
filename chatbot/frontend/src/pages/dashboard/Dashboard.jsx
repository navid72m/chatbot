// src/pages/dashboard/Dashboard.jsx

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  FaUsers, 
  FaShoppingCart, 
  FaMoneyBillWave, 
  FaChartLine,
  FaEye,
  FaEdit,
  FaTrash,
  FaDownload,
  FaPlus
} from 'react-icons/fa';

// Components
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import Chart from '../../components/data/Chart';
import DataTable from '../../components/data/DataTable';
import { useNotification } from '../../context/NotificationContext';

// API
import api from '../../api/client';

const Dashboard = () => {
  const { showSuccess, showError } = useNotification();
  
  // State for stats
  const [statsLoading, setStatsLoading] = useState(true);
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalProducts: 0,
    totalRevenue: 0,
    totalOrders: 0,
  });
  
  // State for revenue chart
  const [revenueChartLoading, setRevenueChartLoading] = useState(true);
  const [revenueChartData, setRevenueChartData] = useState([]);
  const [revenueChartError, setRevenueChartError] = useState(null);
  
  // State for recent orders
  const [ordersLoading, setOrdersLoading] = useState(true);
  const [orders, setOrders] = useState([]);
  
  // Fetch stats data
  useEffect(() => {
    const fetchStats = async () => {
      try {
        setStatsLoading(true);
        
        // This would be your actual API call
        // const response = await api.get('/dashboard/stats');
        
        // Simulated API response for demonstration
        const response = {
          totalUsers: 1248,
          totalProducts: 374,
          totalRevenue: 68429.50,
          totalOrders: 896,
        };
        
        setStats(response);
      } catch (error) {
        console.error('Error fetching stats:', error);
        showError('Failed to load dashboard statistics');
      } finally {
        setStatsLoading(false);
      }
    };
    
    fetchStats();
  }, [showError]);
  
  // Fetch revenue chart data
  useEffect(() => {
    const fetchRevenueData = async () => {
      try {
        setRevenueChartLoading(true);
        setRevenueChartError(null);
        
        // This would be your actual API call
        // const response = await api.get('/dashboard/revenue');
        
        // Simulated API response for demonstration
        const response = [
          { month: 'Jan', revenue: 5240, profit: 2120, orders: 145 },
          { month: 'Feb', revenue: 4800, profit: 1890, orders: 132 },
          { month: 'Mar', revenue: 6540, profit: 2700, orders: 177 },
          { month: 'Apr', revenue: 5780, profit: 2330, orders: 158 },
          { month: 'May', revenue: 7800, profit: 3280, orders: 201 },
          { month: 'Jun', revenue: 9200, profit: 3960, orders: 247 },
          { month: 'Jul', revenue: 8700, profit: 3650, orders: 230 },
          { month: 'Aug', revenue: 9600, profit: 4150, orders: 256 },
          { month: 'Sep', revenue: 8400, profit: 3520, orders: 221 },
          { month: 'Oct', revenue: 7900, profit: 3330, orders: 210 },
          { month: 'Nov', revenue: 9800, profit: 4290, orders: 265 },
          { month: 'Dec', revenue: 12500, profit: 5600, orders: 325 },
        ];
        
        setRevenueChartData(response);
      } catch (error) {
        console.error('Error fetching revenue data:', error);
        setRevenueChartError('Failed to load revenue data');
        showError('Failed to load revenue chart data');
      } finally {
        setRevenueChartLoading(false);
      }
    };
    
    fetchRevenueData();
  }, [showError]);
  
  // Fetch recent orders
  useEffect(() => {
    const fetchOrders = async () => {
      try {
        setOrdersLoading(true);
        
        // This would be your actual API call
        // const response = await api.get('/orders', { params: { limit: 5 } });
        
        // Simulated API response for demonstration
        const response = [
          { id: 'ORD-1234', customer: 'John Doe', date: '2023-12-01', total: 129.99, status: 'Completed' },
          { id: 'ORD-1235', customer: 'Jane Smith', date: '2023-12-02', total: 89.50, status: 'Processing' },
          { id: 'ORD-1236', customer: 'Bob Johnson', date: '2023-12-03', total: 245.75, status: 'Completed' },
          { id: 'ORD-1237', customer: 'Alice Brown', date: '2023-12-03', total: 59.99, status: 'Shipped' },
          { id: 'ORD-1238', customer: 'David Wilson', date: '2023-12-04', total: 179.50, status: 'Processing' },
        ];
        
        setOrders(response);
      } catch (error) {
        console.error('Error fetching orders:', error);
        showError('Failed to load recent orders');
      } finally {
        setOrdersLoading(false);
      }
    };
    
    fetchOrders();
  }, [showError]);
  
  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };
  
  // Dashboard stats cards
  const renderStatsCards = () => {
    return (
      <div className="row">
        <div className="col-md-3 col-sm-6 mb-4">
          <Card
            className="dashboard-stat-card"
            icon={<FaUsers />}
            title={statsLoading ? 'Loading...' : stats.totalUsers.toLocaleString()}
            subtitle="Total Users"
          />
        </div>
        
        <div className="col-md-3 col-sm-6 mb-4">
          <Card
            className="dashboard-stat-card"
            icon={<FaShoppingCart />}
            title={statsLoading ? 'Loading...' : stats.totalProducts.toLocaleString()}
            subtitle="Total Products"
          />
        </div>
        
        <div className="col-md-3 col-sm-6 mb-4">
          <Card
            className="dashboard-stat-card"
            icon={<FaMoneyBillWave />}
            title={statsLoading ? 'Loading...' : formatCurrency(stats.totalRevenue)}
            subtitle="Total Revenue"
          />
        </div>
        
        <div className="col-md-3 col-sm-6 mb-4">
          <Card
            className="dashboard-stat-card"
            icon={<FaChartLine />}
            title={statsLoading ? 'Loading...' : stats.totalOrders.toLocaleString()}
            subtitle="Total Orders"
          />
        </div>
      </div>
    );
  };
  
  // Recent orders table config
  const orderColumns = [
    { field: 'id', header: 'Order ID' },
    { field: 'customer', header: 'Customer' },
    { field: 'date', header: 'Date', render: (value) => new Date(value).toLocaleDateString() },
    { field: 'total', header: 'Total', render: (value) => formatCurrency(value) },
    { 
      field: 'status', 
      header: 'Status',
      render: (value) => {
        let statusClass = '';
        
        switch (value) {
          case 'Completed':
            statusClass = 'success';
            break;
          case 'Processing':
            statusClass = 'warning';
            break;
          case 'Shipped':
            statusClass = 'info';
            break;
          case 'Cancelled':
            statusClass = 'danger';
            break;
          default:
            statusClass = 'secondary';
        }
        
        return <span className={`badge bg-${statusClass}`}>{value}</span>;
      }
    },
  ];
  
  // Table row actions
  const orderRowActions = [
    {
      icon: <FaEye />,
      title: 'View Order',
      buttonClass: 'btn-outline-primary',
      onClick: (row) => {
        // Navigate to order detail
        console.log('View order:', row.id);
      },
    },
    {
      icon: <FaEdit />,
      title: 'Edit Order',
      buttonClass: 'btn-outline-secondary',
      onClick: (row) => {
        // Navigate to edit order
        console.log('Edit order:', row.id);
      },
    },
    {
      icon: <FaTrash />,
      title: 'Delete Order',
      buttonClass: 'btn-outline-danger',
      onClick: (row) => {
        // Show confirmation dialog
        console.log('Delete order:', row.id);
      },
    },
  ];
  
  // Table actions
  const orderTableActions = [
    {
      icon: <FaPlus />,
      label: 'New Order',
      buttonClass: 'btn-primary',
      onClick: () => {
        // Navigate to create order
        console.log('Create new order');
      },
    },
    {
      icon: <FaDownload />,
      label: 'Export',
      buttonClass: 'btn-outline-secondary',
      onClick: () => {
        // Export orders
        console.log('Export orders');
      },
    },
  ];
  
  return (
    <div className="dashboard-page">
      {/* Page header */}
      <div className="page-header mb-4">
        <h1 className="page-title">Dashboard</h1>
        <div className="page-actions">
          <Button variant="primary" icon={<FaDownload />}>
            Download Report
          </Button>
        </div>
      </div>
      
      {/* Stats cards */}
      {renderStatsCards()}
      
      {/* Revenue chart */}
      <div className="row mb-4">
        <div className="col-lg-8 mb-4 mb-lg-0">
          <Chart
            title="Revenue Overview"
            subtitle="Monthly revenue and profit"
            type="line"
            data={revenueChartData}
            xKey="month"
            series={[
              { dataKey: 'revenue', name: 'Revenue' },
              { dataKey: 'profit', name: 'Profit' },
            ]}
            height={350}
            loading={revenueChartLoading}
            error={revenueChartError}
            formatValue={(value) => formatCurrency(value)}
          />
        </div>
        
        <div className="col-lg-4">
          <Chart
            title="Orders by Month"
            type="bar"
            data={revenueChartData}
            xKey="month"
            series={[
              { dataKey: 'orders', name: 'Orders' },
            ]}
            height={350}
            loading={revenueChartLoading}
            error={revenueChartError}
          />
        </div>
      </div>
      
      {/* Recent orders */}
      <div className="row">
        <div className="col-12">
          <DataTable
            title="Recent Orders"
            data={orders}
            columns={orderColumns}
            loading={ordersLoading}
            rowActions={orderRowActions}
            tableActions={orderTableActions}
            pagination={true}
            pageSize={5}
            emptyStateMessage="No recent orders found"
            onRowClick={(row) => console.log('Row clicked:', row.id)}
          />
        </div>
      </div>
      
      {/* Quick links */}
      <div className="row mt-4">
        <div className="col-12">
          <Card title="Quick Links">
            <div className="row">
              <div className="col-md-3 col-sm-6 mb-3">
                <Link to="/users" className="dashboard-quick-link">
                  <div className="dashboard-quick-link-icon">
                    <FaUsers />
                  </div>
                  <span>Manage Users</span>
                </Link>
              </div>
              
              <div className="col-md-3 col-sm-6 mb-3">
                <Link to="/products" className="dashboard-quick-link">
                  <div className="dashboard-quick-link-icon">
                    <FaShoppingCart />
                  </div>
                  <span>View Products</span>
                </Link>
              </div>
              
              <div className="col-md-3 col-sm-6 mb-3">
                <Link to="/orders" className="dashboard-quick-link">
                  <div className="dashboard-quick-link-icon">
                    <FaMoneyBillWave />
                  </div>
                  <span>All Orders</span>
                </Link>
              </div>
              
              <div className="col-md-3 col-sm-6 mb-3">
                <Link to="/analytics" className="dashboard-quick-link">
                  <div className="dashboard-quick-link-icon">
                    <FaChartLine />
                  </div>
                  <span>Analytics</span>
                </Link>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;