// src/components/routing/ProtectedRoute.jsx

import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

/**
 * ProtectedRoute component ensures that only authenticated users can access 
 * certain routes. If a user is not authenticated, they will be redirected 
 * to the login page with the return URL saved in the location state.
 */
const ProtectedRoute = () => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();
  
  // Show loading state if auth state is being determined
  if (loading) {
    return (
      <div className="auth-loading">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }
  
  // If not authenticated, redirect to login with return URL
  if (!isAuthenticated) {
    return (
      <Navigate
        to="/login"
        state={{ from: location }}
        replace
      />
    );
  }
  
  // If authenticated, render the child routes
  return <Outlet />;
};

export default ProtectedRoute;