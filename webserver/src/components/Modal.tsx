import React from 'react';
import './Modal.css';

interface ModalProps {
  show: boolean;
  onClose: () => void;
  title: string | React.ReactNode;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ show, onClose, title, children }) => {
  if (!show) {
    return null;
  }

  return (
    <div className="modal-overlay">
      <div className="modal">
        <div className="modal-header">
          <h2>{title}</h2>
          <button onClick={onClose}>X</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
};

export default Modal;
