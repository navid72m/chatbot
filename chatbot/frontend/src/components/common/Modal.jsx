// src/components/common/Modal.jsx

import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { createPortal } from 'react-dom';
import { FaTimes } from 'react-icons/fa';

// Modal sizes
const MODAL_SIZES = {
  sm: 'modal-sm',
  md: 'modal-md',
  lg: 'modal-lg',
  xl: 'modal-xl',
  fullscreen: 'modal-fullscreen',
};

const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = 'md',
  closeOnClickOutside = true,
  closeOnEsc = true,
  showCloseButton = true,
  className = '',
  modalBodyClassName = '',
  backdropClassName = '',
  centered = false,
  scrollable = true,
  staticBackdrop = false,
  animation = true,
  beforeClose = null,
}) => {
  const modalRef = useRef(null);
  
  // Handle ESC key press
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (closeOnEsc && event.key === 'Escape' && isOpen) {
        handleClose();
      }
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.classList.add('modal-open');
    }
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.classList.remove('modal-open');
    };
  }, [isOpen, closeOnEsc]);
  
  // Handle click outside
  const handleBackdropClick = (event) => {
    if (
      closeOnClickOutside &&
      modalRef.current &&
      !modalRef.current.contains(event.target)
    ) {
      handleClose();
    }
  };
  
  // Handle close button click
  const handleClose = async () => {
    if (typeof beforeClose === 'function') {
      const shouldClose = await beforeClose();
      if (!shouldClose) return;
    }
    
    onClose();
  };
  
  // Don't render if modal is closed
  if (!isOpen) return null;
  
  // Determine class names based on props
  const modalClasses = [
    'modal',
    animation ? 'fade' : '',
    isOpen ? 'show' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');
  
  const dialogClasses = [
    'modal-dialog',
    MODAL_SIZES[size] || MODAL_SIZES.md,
    centered ? 'modal-dialog-centered' : '',
    scrollable ? 'modal-dialog-scrollable' : '',
  ]
    .filter(Boolean)
    .join(' ');
  
  const backdropClasses = [
    'modal-backdrop',
    animation ? 'fade' : '',
    isOpen ? 'show' : '',
    backdropClassName,
  ]
    .filter(Boolean)
    .join(' ');
  
  // Portal for modal
  const modalContent = (
    <>
      <div
        className={backdropClasses}
        onClick={staticBackdrop ? null : handleBackdropClick}
      />
      
      <div
        className={modalClasses}
        tabIndex="-1"
        role="dialog"
        aria-modal="true"
        style={{ display: 'block' }}
      >
        <div className={dialogClasses} ref={modalRef}>
          <div className="modal-content">
            {/* Modal Header */}
            {(title || showCloseButton) && (
              <div className="modal-header">
                {title && <h5 className="modal-title">{title}</h5>}
                
                {showCloseButton && (
                  <button
                    type="button"
                    className="btn-close"
                    aria-label="Close"
                    onClick={handleClose}
                  >
                    <FaTimes />
                  </button>
                )}
              </div>
            )}
            
            {/* Modal Body */}
            <div className={`modal-body ${modalBodyClassName}`}>{children}</div>
            
            {/* Modal Footer */}
            {footer && <div className="modal-footer">{footer}</div>}
          </div>
        </div>
      </div>
    </>
  );
  
  return createPortal(modalContent, document.body);
};

Modal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.node,
  children: PropTypes.node.isRequired,
  footer: PropTypes.node,
  size: PropTypes.oneOf(['sm', 'md', 'lg', 'xl', 'fullscreen']),
  closeOnClickOutside: PropTypes.bool,
  closeOnEsc: PropTypes.bool,
  showCloseButton: PropTypes.bool,
  className: PropTypes.string,
  modalBodyClassName: PropTypes.string,
  backdropClassName: PropTypes.string,
  centered: PropTypes.bool,
  scrollable: PropTypes.bool,
  staticBackdrop: PropTypes.bool,
  animation: PropTypes.bool,
  beforeClose: PropTypes.func,
};

export default Modal;