/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { memo } from 'react';
import { motion, AnimatePresence } from 'motion/react';

interface CursorData {
  userId: string;
  userName: string;
  color: string;
  x: number;
  y: number;
}

interface CollaboratorCursorsProps {
  cursors: Map<string, CursorData>;
}

const Cursor = memo(({ cursor }: { cursor: CursorData }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.5 }}
      transition={{ duration: 0.15 }}
      className="absolute pointer-events-none z-50"
      style={{
        left: cursor.x,
        top: cursor.y,
        transform: 'translate(-2px, -2px)',
      }}
    >
      {/* Cursor SVG */}
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.3))' }}
      >
        <path
          d="M5.65376 12.4563L6.95161 6.90473C7.44052 4.86938 7.68498 3.85171 8.28903 3.52955C8.81295 3.25057 9.44042 3.24853 9.96618 3.52428C10.5729 3.84165 10.8242 4.85735 11.327 6.88876L16.2766 17.9999C16.8109 19.1418 17.0781 19.7128 16.9519 20.1235C16.8416 20.4796 16.5845 20.7709 16.2427 20.9248C15.8491 21.1011 15.2587 20.9127 14.0778 20.536L11.5348 19.715C10.9767 19.5351 10.6976 19.4452 10.4155 19.4235C10.1642 19.4041 9.91161 19.4237 9.66671 19.4815C9.39046 19.5466 9.12667 19.6779 8.59909 19.9407L5.18762 21.6399C4.1052 22.1794 3.56399 22.4492 3.14871 22.3535C2.78819 22.2703 2.48576 22.0203 2.3356 21.6783C2.16161 21.2833 2.32703 20.7072 2.65786 19.5551L5.65376 12.4563Z"
          fill={cursor.color}
          stroke="white"
          strokeWidth="1.5"
        />
      </svg>

      {/* User name label */}
      <div
        className="absolute left-4 top-4 px-2 py-0.5 rounded text-xs font-medium text-white whitespace-nowrap"
        style={{ backgroundColor: cursor.color }}
      >
        {cursor.userName}
      </div>
    </motion.div>
  );
});

Cursor.displayName = 'Cursor';

export const CollaboratorCursors = memo(({ cursors }: CollaboratorCursorsProps) => {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <AnimatePresence>
        {Array.from(cursors.entries()).map(([userId, cursor]) => (
          <Cursor key={userId} cursor={cursor} />
        ))}
      </AnimatePresence>
    </div>
  );
});

CollaboratorCursors.displayName = 'CollaboratorCursors';
