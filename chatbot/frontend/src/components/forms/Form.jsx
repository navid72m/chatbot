// src/components/forms/Form.jsx

import React, { useState, useCallback, useEffect } from 'react';
import PropTypes from 'prop-types';
import { FormProvider } from './FormContext';

const Form = ({
  initialValues = {},
  validationSchema = null,
  onSubmit,
  children,
  resetOnSubmit = false,
  className = '',
}) => {
  // State for form values, errors, touched fields, and submission status
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  
  // Reset form when initialValues change (if component is reused)
  useEffect(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitted(false);
  }, [initialValues]);
  
  // Validate single field
  const validateField = useCallback(
    (name, value) => {
      if (!validationSchema) return '';
      
      try {
        // Check if field exists in validation schema
        const fieldSchema = validationSchema.fields[name];
        if (!fieldSchema) return '';
        
        // Validate field
        fieldSchema.validateSync(value);
        return '';
      } catch (error) {
        return error.message;
      }
    },
    [validationSchema]
  );
  
  // Validate all fields
  const validateForm = useCallback(() => {
    if (!validationSchema) return {};
    
    const newErrors = {};
    
    // Validate each field
    Object.keys(values).forEach((name) => {
      const error = validateField(name, values[name]);
      if (error) {
        newErrors[name] = error;
      }
    });
    
    return newErrors;
  }, [validationSchema, validateField, values]);
  
  // Handle field change
  const handleChange = useCallback(
    (event) => {
      const { name, value, type, checked } = event.target;
      
      // Handle different input types
      const newValue = type === 'checkbox' ? checked : value;
      
      setValues((prevValues) => ({
        ...prevValues,
        [name]: newValue,
      }));
      
      // Validate field if touched
      if (touched[name]) {
        const error = validateField(name, newValue);
        setErrors((prevErrors) => ({
          ...prevErrors,
          [name]: error,
        }));
      }
    },
    [touched, validateField]
  );
  
  // Handle field blur
  const handleBlur = useCallback(
    (event) => {
      const { name, value } = event.target;
      
      // Mark field as touched
      setTouched((prevTouched) => ({
        ...prevTouched,
        [name]: true,
      }));
      
      // Validate field
      const error = validateField(name, value);
      setErrors((prevErrors) => ({
        ...prevErrors,
        [name]: error,
      }));
    },
    [validateField]
  );
  
  // Set field value programmatically
  const setFieldValue = useCallback(
    (name, value) => {
      setValues((prevValues) => ({
        ...prevValues,
        [name]: value,
      }));
      
      // Validate field if touched
      if (touched[name]) {
        const error = validateField(name, value);
        setErrors((prevErrors) => ({
          ...prevErrors,
          [name]: error,
        }));
      }
    },
    [touched, validateField]
  );
  
  // Set multiple field values programmatically
  const setFieldValues = useCallback(
    (newValues) => {
      setValues((prevValues) => ({
        ...prevValues,
        ...newValues,
      }));
      
      // Validate touched fields
      Object.keys(newValues).forEach((name) => {
        if (touched[name]) {
          const error = validateField(name, newValues[name]);
          setErrors((prevErrors) => ({
            ...prevErrors,
            [name]: error,
          }));
        }
      });
    },
    [touched, validateField]
  );
  
  // Reset form to initial values
  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitted(false);
  }, [initialValues]);
  
  // Handle form submission
  const handleSubmit = useCallback(
    async (event) => {
      event.preventDefault();
      
      // Validate all fields
      const formErrors = validateForm();
      setErrors(formErrors);
      
      // Mark all fields as touched
      const touchedFields = {};
      Object.keys(values).forEach((key) => {
        touchedFields[key] = true;
      });
      setTouched(touchedFields);
      
      // Check if form is valid
      const isValid = Object.keys(formErrors).length === 0;
      
      if (isValid) {
        setIsSubmitting(true);
        setIsSubmitted(true);
        
        try {
          // Call onSubmit handler with form values
          await onSubmit(values);
          
          // Reset form if needed
          if (resetOnSubmit) {
            resetForm();
          }
        } catch (error) {
          console.error('Form submission error:', error);
          // You could set specific field errors if the API returns them
        } finally {
          setIsSubmitting(false);
        }
      }
    },
    [onSubmit, resetOnSubmit, resetForm, validateForm, values]
  );
  
  // Create form context value for children
  const formContextValue = {
    values,
    errors,
    touched,
    isSubmitting,
    isSubmitted,
    handleChange,
    handleBlur,
    setFieldValue,
    setFieldValues,
    resetForm,
  };
  
  // Render children with form context
  return (
    <form onSubmit={handleSubmit} className={className} noValidate>
      <FormProvider value={formContextValue}>
        {typeof children === 'function'
          ? children(formContextValue)
          : children}
      </FormProvider>
    </form>
  );
};

Form.propTypes = {
  initialValues: PropTypes.object,
  validationSchema: PropTypes.object,
  onSubmit: PropTypes.func.isRequired,
  children: PropTypes.oneOfType([PropTypes.func, PropTypes.node]).isRequired,
  resetOnSubmit: PropTypes.bool,
  className: PropTypes.string,
};

export default Form;