// src/components/common/Button.jsx

import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';

const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  type = 'button',
  disabled = false,
  loading = false,
  icon = null,
  iconPosition = 'left',
  block = false,
  rounded = false,
  outline = false,
  className = '',
  onClick,
  href,
  to,
  target,
  rel,
  title,
  ariaLabel,
  ...rest
}) => {
  // Determine button classes
  const buttonClasses = [
    'btn',
    `btn-${outline ? 'outline-' : ''}${variant}`,
    `btn-${size}`,
    block ? 'btn-block' : '',
    rounded ? 'btn-rounded' : '',
    loading ? 'btn-loading' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Content with optional icon
  const content = (
    <>
      {loading && <span className="btn-spinner"></span>}
      
      {icon && iconPosition === 'left' && !loading && (
        <span className="btn-icon btn-icon-left">{icon}</span>
      )}
      
      {children && <span className="btn-text">{children}</span>}
      
      {icon && iconPosition === 'right' && !loading && (
        <span className="btn-icon btn-icon-right">{icon}</span>
      )}
    </>
  );
  
  // If href is provided, render as anchor
  if (href) {
    return (
      <a
        href={href}
        className={buttonClasses}
        target={target}
        rel={rel || (target === '_blank' ? 'noopener noreferrer' : undefined)}
        aria-label={ariaLabel}
        title={title}
        onClick={onClick}
        {...rest}
      >
        {content}
      </a>
    );
  }
  
  // If to is provided, render as Link
  if (to) {
    return (
      <Link
        to={to}
        className={buttonClasses}
        aria-label={ariaLabel}
        title={title}
        onClick={onClick}
        {...rest}
      >
        {content}
      </Link>
    );
  }
  
  // Otherwise, render as button
  return (
    <button
      type={type}
      className={buttonClasses}
      disabled={disabled || loading}
      aria-label={ariaLabel}
      title={title}
      onClick={onClick}
      {...rest}
    >
      {content}
    </button>
  );
};

Button.propTypes = {
  children: PropTypes.node,
  variant: PropTypes.oneOf([
    'primary',
    'secondary',
    'success',
    'danger',
    'warning',
    'info',
    'light',
    'dark',
    'link',
  ]),
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  type: PropTypes.oneOf(['button', 'submit', 'reset']),
  disabled: PropTypes.bool,
  loading: PropTypes.bool,
  icon: PropTypes.node,
  iconPosition: PropTypes.oneOf(['left', 'right']),
  block: PropTypes.bool,
  rounded: PropTypes.bool,
  outline: PropTypes.bool,
  className: PropTypes.string,
  onClick: PropTypes.func,
  href: PropTypes.string,
  to: PropTypes.oneOfType([PropTypes.string, PropTypes.object]),
  target: PropTypes.string,
  rel: PropTypes.string,
  title: PropTypes.string,
  ariaLabel: PropTypes.string,
};

export default Button;