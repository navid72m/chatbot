// src/components/layout/AppLayout.jsx

import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { 
  FaBars, 
  FaUser, 
  FaCog, 
  FaSignOutAlt, 
  FaBell, 
  FaEnvelope, 
  FaSearch,
  FaTachometerAlt,
  FaChartBar,
  FaUsers,
  FaBox,
  FaFileAlt,
  FaClipboardList,
  FaQuestionCircle,
  FaSun,
  FaMoon,
  FaChevronDown
} from 'react-icons/fa';

import { useAuth } from '../../context/AuthContext';

const AppLayout = ({ children }) => {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  
  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };
  
  // Toggle user menu
  const toggleUserMenu = () => {
    setUserMenuOpen(!userMenuOpen);
  };
  
  // Toggle notifications
  const toggleNotifications = () => {
    setNotificationsOpen(!notificationsOpen);
  };
  
  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark-mode');
  };
  
  // Handle logout
  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };
  
  // Navigation items
  const navItems = [
    {
      label: 'Dashboard',
      icon: <FaTachometerAlt />,
      path: '/dashboard',
    },
    {
      label: 'Analytics',
      icon: <FaChartBar />,
      path: '/analytics',
    },
    {
      label: 'User Management',
      icon: <FaUsers />,
      path: '/users',
      subItems: [
        { label: 'Users', path: '/users' },
        { label: 'Roles', path: '/roles' },
        { label: 'Permissions', path: '/permissions' },
      ],
    },
    {
      label: 'Products',
      icon: <FaBox />,
      path: '/products',
    },
    {
      label: 'Reports',
      icon: <FaFileAlt />,
      path: '/reports',
    },
    {
      label: 'Tasks',
      icon: <FaClipboardList />,
      path: '/tasks',
    },
    {
      label: 'Settings',
      icon: <FaCog />,
      path: '/settings',
      subItems: [
        { label: 'General', path: '/settings/general' },
        { label: 'Notifications', path: '/settings/notifications' },
        { label: 'Security', path: '/settings/security' },
        { label: 'Integrations', path: '/settings/integrations' },
      ],
    },
    {
      label: 'Help',
      icon: <FaQuestionCircle />,
      path: '/help',
    },
  ];
  
  // Check if path is active
  const isActive = (path) => {
    return location.pathname === path || location.pathname.startsWith(`${path}/`);
  };
  
  // Render navigation items
  const renderNavItems = (items) => {
    return items.map((item) => {
      const active = isActive(item.path);
      const hasSubItems = item.subItems && item.subItems.length > 0;
      const [subMenuOpen, setSubMenuOpen] = useState(active);
      
      return (
        <li key={item.path} className={`nav-item ${active ? 'active' : ''}`}>
          {hasSubItems ? (
            <>
              <button
                className={`nav-link ${active ? 'active' : ''}`}
                onClick={() => setSubMenuOpen(!subMenuOpen)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className={`nav-text ${sidebarCollapsed ? 'd-none' : ''}`}>
                  {item.label}
                </span>
                <span className={`nav-chevron ${sidebarCollapsed ? 'd-none' : ''}`}>
                  <FaChevronDown
                    className={`transition-transform ${
                      subMenuOpen ? 'rotate-180' : ''
                    }`}
                  />
                </span>
              </button>
              
              <ul className={`nav-submenu ${subMenuOpen ? 'open' : ''}`}>
                {item.subItems.map((subItem) => (
                  <li key={subItem.path} className="nav-submenu-item">
                    <Link
                      to={subItem.path}
                      className={`nav-submenu-link ${
                        isActive(subItem.path) ? 'active' : ''
                      }`}
                    >
                      <span className="nav-submenu-text">{subItem.label}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <Link to={item.path} className={`nav-link ${active ? 'active' : ''}`}>
              <span className="nav-icon">{item.icon}</span>
              <span className={`nav-text ${sidebarCollapsed ? 'd-none' : ''}`}>
                {item.label}
              </span>
            </Link>
          )}
        </li>
      );
    });
  };

  return (
    <div className={`app-container ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <Link to="/dashboard" className="logo">
            <img src="/logo.svg" alt="Logo" className="logo-img" />
            <span className={`logo-text ${sidebarCollapsed ? 'd-none' : ''}`}>
              Admin Dashboard
            </span>
          </Link>
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            <FaBars />
          </button>
        </div>
        
        <nav className="sidebar-nav">
          <ul className="nav-list">{renderNavItems(navItems)}</ul>
        </nav>
      </aside>
      
      {/* Main content */}
      <main className="main-content">
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <button className="sidebar-toggle d-md-none" onClick={toggleSidebar}>
              <FaBars />
            </button>
            
            <div className="search-container">
              <div className="search-box">
                <FaSearch className="search-icon" />
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search..."
                />
              </div>
            </div>
          </div>
          
          <div className="header-right">
            {/* Dark mode toggle */}
            <button className="header-icon-button" onClick={toggleDarkMode}>
              {darkMode ? <FaSun /> : <FaMoon />}
            </button>
            
            {/* Notifications */}
            <div className="dropdown">
              <button
                className="header-icon-button"
                onClick={toggleNotifications}
              >
                <FaBell />
                <span className="badge">3</span>
              </button>
              
              {notificationsOpen && (
                <div className="dropdown-menu">
                  <div className="dropdown-header">
                    <h6>Notifications</h6>
                    <button className="dropdown-link">Mark all as read</button>
                  </div>
                  
                  <div className="dropdown-body">
                    <div className="notification-item unread">
                      <div className="notification-icon success">
                        <FaEnvelope />
                      </div>
                      <div className="notification-content">
                        <p>You have a new message</p>
                        <span className="notification-time">1 hour ago</span>
                      </div>
                    </div>
                    
                    <div className="notification-item unread">
                      <div className="notification-icon warning">
                        <FaClipboardList />
                      </div>
                      <div className="notification-content">
                        <p>Task deadline approaching</p>
                        <span className="notification-time">2 hours ago</span>
                      </div>
                    </div>
                    
                    <div className="notification-item">
                      <div className="notification-icon info">
                        <FaUsers />
                      </div>
                      <div className="notification-content">
                        <p>New user registered</p>
                        <span className="notification-time">Yesterday</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="dropdown-footer">
                    <Link to="/notifications" className="dropdown-link">
                      View all notifications
                    </Link>
                  </div>
                </div>
              )}
            </div>
            
            {/* User menu */}
            <div className="dropdown">
              <button className="user-menu-button" onClick={toggleUserMenu}>
                <img
                  src={user?.avatar || '/default-avatar.png'}
                  alt="User avatar"
                  className="user-avatar"
                />
                <span className="user-name d-none d-md-inline">
                  {user?.name || 'User'}
                </span>
              </button>
              
              {userMenuOpen && (
                <div className="dropdown-menu user-dropdown">
                  <div className="dropdown-header user-dropdown-header">
                    <img
                      src={user?.avatar || '/default-avatar.png'}
                      alt="User avatar"
                      className="user-avatar-lg"
                    />
                    <div>
                      <h6>{user?.name || 'User'}</h6>
                      <p>{user?.email || 'user@example.com'}</p>
                    </div>
                  </div>
                  
                  <div className="dropdown-body">
                    <Link to="/profile" className="dropdown-item">
                      <FaUser className="dropdown-item-icon" />
                      <span>My Profile</span>
                    </Link>
                    
                    <Link to="/settings" className="dropdown-item">
                      <FaCog className="dropdown-item-icon" />
                      <span>Settings</span>
                    </Link>
                    
                    <button className="dropdown-item" onClick={handleLogout}>
                      <FaSignOutAlt className="dropdown-item-icon" />
                      <span>Logout</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>
        
        {/* Page content */}
        <div className="page-container">
          {children || <Outlet />}
        </div>
      </main>
    </div>
  );
};

AppLayout.propTypes = {
  children: PropTypes.node,
};

export default AppLayout;