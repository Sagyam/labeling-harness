import React from 'react'

export interface ToastMessage {
  id: string
  text: string
  type?: 'success' | 'info' | 'warning' | 'error'
}

interface ToastContainerProps {
  toasts: ToastMessage[]
  onDismiss: (id: string) => void
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast toast-${toast.type || 'info'}`}
          onClick={() => onDismiss(toast.id)}
        >
          <span>{toast.text}</span>
        </div>
      ))}
    </div>
  )
}
