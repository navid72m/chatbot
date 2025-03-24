// src/components/common/Input.jsx

import React, { forwardRef } from 'react';
import PropTypes from 'prop-types';

const Input = forwardRef(({
  type = 'text',
  id,
  name,
  value,
  placeholder,
  onChange,
  onBlur,
  onFocus,
  disabled = false,
  readOnly = false,
  required = false,
  autoFocus = false,
  autoComplete = 'off',
  min,
  max,
  step,
  pattern,
  label,
  helperText,
  error,
  success,
  size = 'md',
  prefix,
  suffix,
  className = '',
  inputClassName = '',
  labelClassName = '',
  helperTextClassName = '',
  errorClassName = '',
  successClassName = '',
  containerClassName = '',
  ...rest
}, ref) => {
  // Determine container classes
  const containerClasses = [
    'input-container',
    error ? 'has-error' : '',
    success ? 'has-success' : '',
    disabled ? 'is-disabled' : '',
    readOnly ? 'is-readonly' : '',
    prefix ? 'has-prefix' : '',
    suffix ? 'has-suffix' : '',
    `input-${size}`,
    containerClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Determine input classes
  const inputClasses = [
    'input',
    error ? 'input-error' : '',
    success ? 'input-success' : '',
    inputClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Determine helper text classes
  const helperTextClasses = [
    'input-helper-text',
    error ? 'input-helper-text-error' : '',
    success ? 'input-helper-text-success' : '',
    helperTextClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Input element
  const inputElement = (
    <div className={containerClasses}>
      {prefix && <div className="input-prefix">{prefix}</div>}
      
      <input
        ref={ref}
        type={type}
        id={id || name}
        name={name}
        value={value}
        className={inputClasses}
        placeholder={placeholder}
        disabled={disabled}
        readOnly={readOnly}
        required={required}
        autoFocus={autoFocus}
        autoComplete={autoComplete}
        onChange={onChange}
        onBlur={onBlur}
        onFocus={onFocus}
        min={min}
        max={max}
        step={step}
        pattern={pattern}
        {...rest}
      />
      
      {suffix && <div className="input-suffix">{suffix}</div>}
    </div>
  );
  
  return (
    <div className={`input-group ${className}`}>
      {label && (
        <label
          htmlFor={id || name}
          className={`input-label ${labelClassName}`}
        >
          {label}
          {required && <span className="input-required">*</span>}
        </label>
      )}
      
      {inputElement}
      
      {(helperText || error || success) && (
        <div
          className={
            error
              ? `input-message input-error-message ${errorClassName}`
              : success
                ? `input-message input-success-message ${successClassName}`
                : helperTextClasses
          }
        >
          {error || success || helperText}
        </div>
      )}
    </div>
  );
});

Input.displayName = 'Input';

Input.propTypes = {
  type: PropTypes.string,
  id: PropTypes.string,
  name: PropTypes.string,
  value: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
  ]),
  placeholder: PropTypes.string,
  onChange: PropTypes.func,
  onBlur: PropTypes.func,
  onFocus: PropTypes.func,
  disabled: PropTypes.bool,
  readOnly: PropTypes.bool,
  required: PropTypes.bool,
  autoFocus: PropTypes.bool,
  autoComplete: PropTypes.string,
  min: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  max: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  step: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  pattern: PropTypes.string,
  label: PropTypes.node,
  helperText: PropTypes.node,
  error: PropTypes.node,
  success: PropTypes.node,
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  prefix: PropTypes.node,
  suffix: PropTypes.node,
  className: PropTypes.string,
  inputClassName: PropTypes.string,
  labelClassName: PropTypes.string,
  helperTextClassName: PropTypes.string,
  errorClassName: PropTypes.string,
  successClassName: PropTypes.string,
  containerClassName: PropTypes.string,
};

export default Input;