/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Users,
  Copy,
  Check,
  Link2,
  LogOut,
  Loader2,
  AlertCircle,
  UserPlus,
  Globe,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useUIStore } from '@/stores/uiStore';
import { useCollaborationStore } from '@/stores/collaborationStore';
import { useAuthStore } from '@/stores/authStore';
import { collaborationManager } from '@/lib/collaboration/supabaseRealtime';
import { cn } from '@/lib/utils/cn';

type ModalView = 'main' | 'join';

export function CollaborationModal() {
  const { t } = useTranslation();
  const { isCollaborationModalOpen, closeCollaborationModal } = useUIStore();
  const { profile } = useAuthStore();
  const {
    sessionId,
    collaborators,
    connectionStatus,
    setSessionId,
    setCollaborators,
    setConnectionStatus,
    reset,
  } = useCollaborationStore();

  const [view, setView] = useState<ModalView>('main');
  const [joinRoomId, setJoinRoomId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const isConnected = connectionStatus === 'connected';
  const userName = profile ? `${profile.firstName} ${profile.lastName}` : 'Anonymous';

  // Set up collaboration manager callbacks
  useEffect(() => {
    collaborationManager.onStatusChange = (status) => {
      setConnectionStatus(status);
    };

    collaborationManager.onPeersChange = (peers) => {
      setCollaborators(peers);
    };

    collaborationManager.onError = (errorMsg) => {
      setError(errorMsg);
    };

    return () => {
      collaborationManager.onStatusChange = null;
      collaborationManager.onPeersChange = null;
      collaborationManager.onError = null;
    };
  }, [setConnectionStatus, setCollaborators]);

  const handleStartSession = async () => {
    setIsLoading(true);
    setError(null);

    const result = await collaborationManager.startSession(userName);

    if (result.success && result.roomId) {
      setSessionId(result.roomId);
    } else {
      setError(result.error || t('collaboration.errors.startFailed'));
    }

    setIsLoading(false);
  };

  const handleJoinSession = async () => {
    if (!joinRoomId.trim()) {
      setError(t('collaboration.errors.roomIdRequired'));
      return;
    }

    setIsLoading(true);
    setError(null);

    const result = await collaborationManager.joinSession(joinRoomId.trim(), userName);

    if (result.success) {
      setSessionId(joinRoomId.trim());
      setView('main');
      setJoinRoomId('');
    } else {
      setError(result.message || t('collaboration.errors.joinFailed'));
    }

    setIsLoading(false);
  };

  const handleLeaveSession = async () => {
    setIsLoading(true);
    await collaborationManager.disconnect();
    reset();
    setIsLoading(false);
  };

  const handleCopyLink = async () => {
    if (!sessionId) return;

    const link = `${window.location.origin}/editor?collab=${sessionId}`;
    await navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyRoomId = async () => {
    if (!sessionId) return;

    await navigator.clipboard.writeText(sessionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isCollaborationModalOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
        onClick={closeCollaborationModal}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.2 }}
        className="relative z-10 w-full max-w-md mx-4"
      >
        <div className="bg-bg-secondary rounded-2xl border border-border-default shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-border-default bg-gradient-to-b from-soft-violet/5 to-transparent flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-soft-violet/10 rounded-lg">
                <Users size={20} className="text-soft-violet" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-text-primary">
                  {t('collaboration.title')}
                </h2>
                <p className="text-small text-text-muted">
                  {isConnected
                    ? t('collaboration.connectedTo', { roomId: sessionId })
                    : t('collaboration.subtitle')}
                </p>
              </div>
            </div>
            <button
              onClick={closeCollaborationModal}
              className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            <AnimatePresence mode="wait">
              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mb-4 p-3 rounded-lg bg-warm-coral/10 border border-warm-coral/20 flex items-start gap-2"
                >
                  <AlertCircle size={18} className="text-warm-coral flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-warm-coral">{error}</p>
                </motion.div>
              )}

              {isConnected ? (
                /* Connected View */
                <motion.div
                  key="connected"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {/* Room Info */}
                  <div className="p-4 bg-bg-tertiary rounded-xl">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-small text-text-muted">{t('collaboration.roomId')}</span>
                      <div className="flex items-center gap-1">
                        <span className="w-2 h-2 bg-fresh-mint rounded-full animate-pulse" />
                        <span className="text-small text-fresh-mint">{t('collaboration.connected')}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 px-3 py-2 bg-bg-primary rounded-lg text-text-primary font-mono text-sm">
                        {sessionId}
                      </code>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleCopyRoomId}
                        className="flex-shrink-0"
                      >
                        {copied ? <Check size={16} className="text-fresh-mint" /> : <Copy size={16} />}
                      </Button>
                    </div>
                  </div>

                  {/* Copy Link Button */}
                  <Button
                    variant="secondary"
                    size="md"
                    onClick={handleCopyLink}
                    leftIcon={<Link2 size={16} />}
                    className="w-full"
                  >
                    {copied ? t('collaboration.linkCopied') : t('collaboration.copyLink')}
                  </Button>

                  {/* Collaborators List */}
                  <div>
                    <h3 className="text-small font-medium text-text-secondary mb-2">
                      {t('collaboration.participants')} ({collaborators.length}/3)
                    </h3>
                    <div className="space-y-2">
                      {collaborators.map((collab) => (
                        <div
                          key={collab.id}
                          className="flex items-center gap-3 p-2 rounded-lg bg-bg-tertiary"
                        >
                          <div
                            className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                            style={{ backgroundColor: collab.color }}
                          >
                            {collab.name.charAt(0).toUpperCase()}
                          </div>
                          <span className="text-text-primary">{collab.name}</span>
                          {collab.isOnline && (
                            <span className="ml-auto w-2 h-2 bg-fresh-mint rounded-full" />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Leave Button */}
                  <Button
                    variant="danger"
                    size="md"
                    onClick={handleLeaveSession}
                    isLoading={isLoading}
                    leftIcon={<LogOut size={16} />}
                    className="w-full"
                  >
                    {t('collaboration.leave')}
                  </Button>
                </motion.div>
              ) : view === 'join' ? (
                /* Join Session View */
                <motion.div
                  key="join"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  <Input
                    label={t('collaboration.enterRoomId')}
                    placeholder="abc12345"
                    value={joinRoomId}
                    onChange={(e) => setJoinRoomId(e.target.value)}
                    leftIcon={<Globe size={18} />}
                  />

                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="md"
                      onClick={() => {
                        setView('main');
                        setError(null);
                      }}
                      className="flex-1"
                    >
                      {t('common.cancel')}
                    </Button>
                    <Button
                      variant="primary"
                      size="md"
                      onClick={handleJoinSession}
                      isLoading={isLoading}
                      className="flex-1"
                    >
                      {t('collaboration.join')}
                    </Button>
                  </div>
                </motion.div>
              ) : (
                /* Main View - Not Connected */
                <motion.div
                  key="main"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {/* Info Card */}
                  <div className="p-4 bg-electric-indigo/5 border border-electric-indigo/20 rounded-xl">
                    <h3 className="text-sm font-medium text-text-primary mb-2">
                      {t('collaboration.howItWorks')}
                    </h3>
                    <ul className="text-small text-text-secondary space-y-1">
                      <li>{t('collaboration.info1')}</li>
                      <li>{t('collaboration.info2')}</li>
                      <li>{t('collaboration.info3')}</li>
                    </ul>
                  </div>

                  {/* Start Session */}
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={handleStartSession}
                    isLoading={isLoading}
                    leftIcon={<Users size={18} />}
                    className="w-full"
                  >
                    {t('collaboration.startSession')}
                  </Button>

                  {/* Divider */}
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-px bg-border-default" />
                    <span className="text-small text-text-muted">{t('collaboration.or')}</span>
                    <div className="flex-1 h-px bg-border-default" />
                  </div>

                  {/* Join Session */}
                  <Button
                    variant="secondary"
                    size="lg"
                    onClick={() => {
                      setView('join');
                      setError(null);
                    }}
                    leftIcon={<UserPlus size={18} />}
                    className="w-full"
                  >
                    {t('collaboration.joinSession')}
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Footer */}
          <div className="px-6 py-3 border-t border-border-default bg-bg-tertiary/50">
            <p className="text-small text-text-muted text-center">
              {t('collaboration.privacyNote')}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
