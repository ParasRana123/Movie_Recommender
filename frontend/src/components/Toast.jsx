import React from 'react';
import { useWatchlist } from '../context/WatchlistContext';

export default function Toast() {
  const { toast, undoRemove } = useWatchlist();

  if (!toast.visible) return null;

  return (
    <div className="toast-notification">
      <span>{toast.message}</span>
      {toast.canUndo && (
        <button className="toast-undo-btn" onClick={undoRemove}>
          UNDO
        </button>
      )}
    </div>
  );
}
