// src/components/forms/FormContext.jsx

import React, { createContext } from 'react';

// Create a context for form state and methods
export const FormContext = createContext({
  values: {},
  errors: {},
  touched: {},
  isSubmitting: false,
  isSubmitted: false,
  handleChange: () => {},
  handleBlur: () => {},
  setFieldValue: () => {},
  setFieldValues: () => {},
  resetForm: () => {},
});

// Form Provider component
export const FormProvider = ({ children, value }) => {
  return (
    <FormContext.Provider value={value}>
      {children}
    </FormContext.Provider>
  );
};

export default FormContext;