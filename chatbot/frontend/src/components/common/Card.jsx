// src/components/common/Card.jsx

import React from 'react';
import PropTypes from 'prop-types';

const Card = ({
  children,
  title,
  subtitle,
  icon,
  headerAction,
  footer,
  className = '',
  headerClassName = '',
  bodyClassName = '',
  footerClassName = '',
  noPadding = false,
  bordered = true,
  shadow = true,
  hover = false,
}) => {
  // Base card class
  const cardClasses = [
    'card',
    bordered ? 'card-bordered' : '',
    shadow ? 'card-shadow' : '',
    hover ? 'card-hover' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Header classes
  const headerClasses = [
    'card-header',
    headerClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Body classes
  const bodyClasses = [
    'card-body',
    noPadding ? 'p-0' : '',
    bodyClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Footer classes
  const footerClasses = [
    'card-footer',
    footerClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Render card header if any header-related props are provided
  const hasHeader = title || subtitle || icon || headerAction;
  
  return (
    <div className={cardClasses}>
      {hasHeader && (
        <div className={headerClasses}>
          <div className="card-header-content">
            {icon && <div className="card-icon">{icon}</div>}
            
            <div className="card-titles">
              {title && (
                typeof title === 'string'
                  ? <h5 className="card-title">{title}</h5>
                  : title
              )}
              
              {subtitle && (
                typeof subtitle === 'string'
                  ? <p className="card-subtitle">{subtitle}</p>
                  : subtitle
              )}
            </div>
          </div>
          
          {headerAction && (
            <div className="card-header-action">
              {headerAction}
            </div>
          )}
        </div>
      )}
      
      <div className={bodyClasses}>{children}</div>
      
      {footer && <div className={footerClasses}>{footer}</div>}
    </div>
  );
};

Card.propTypes = {
  children: PropTypes.node.isRequired,
  title: PropTypes.node,
  subtitle: PropTypes.node,
  icon: PropTypes.node,
  headerAction: PropTypes.node,
  footer: PropTypes.node,
  className: PropTypes.string,
  headerClassName: PropTypes.string,
  bodyClassName: PropTypes.string,
  footerClassName: PropTypes.string,
  noPadding: PropTypes.bool,
  bordered: PropTypes.bool,
  shadow: PropTypes.bool,
  hover: PropTypes.bool,
};

export default Card;