/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { Lock, ArrowRight, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAuthStore } from '@/stores/authStore';

export default function ResetPasswordPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const { updatePassword, isLoading } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!password.trim()) {
      setError(t('auth.errors.passwordRequired'));
      return;
    }

    if (password.length < 6) {
      setError(t('auth.errors.passwordLength'));
      return;
    }

    if (password !== confirmPassword) {
      setError(t('auth.errors.passwordMismatch'));
      return;
    }

    const result = await updatePassword(password);
    if (result.success) {
      setSuccess(true);
      setTimeout(() => {
        navigate('/editor');
      }, 2000);
    } else {
      setError(result.error || t('auth.errors.updateFailed'));
    }
  };

  return (
    <div className="min-h-screen bg-deep-navy flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-md"
      >
        <div className="bg-bg-secondary rounded-2xl border border-border-default shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="px-8 pt-8 pb-6 text-center border-b border-border-default bg-gradient-to-b from-electric-indigo/5 to-transparent">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-14 h-14 rounded-xl mx-auto mb-4 shadow-glow" />
            <h1 className="text-2xl font-bold text-text-primary">
              {t('auth.resetPassword')}
            </h1>
            <p className="text-text-secondary mt-2">
              {t('auth.resetPasswordSubtitle')}
            </p>
          </div>

          {/* Form */}
          {success ? (
            <div className="p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-16 h-16 mx-auto mb-4 rounded-full bg-fresh-mint/10 flex items-center justify-center"
              >
                <CheckCircle size={32} className="text-fresh-mint" />
              </motion.div>
              <h2 className="text-lg font-semibold text-text-primary mb-2">
                {t('auth.passwordUpdated')}
              </h2>
              <p className="text-text-secondary text-sm">
                {t('auth.redirecting')}
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="p-8 space-y-4">
              <Input
                label={t('auth.newPassword')}
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                leftIcon={<Lock size={18} />}
                disabled={isLoading}
              />

              <Input
                label={t('auth.confirmPassword')}
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                leftIcon={<Lock size={18} />}
                disabled={isLoading}
              />

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-lg bg-warm-coral/10 border border-warm-coral/20"
                >
                  <p className="text-sm text-warm-coral">{error}</p>
                </motion.div>
              )}

              <Button
                type="submit"
                variant="primary"
                size="lg"
                isLoading={isLoading}
                className="w-full mt-6"
                rightIcon={!isLoading ? <ArrowRight size={18} /> : undefined}
              >
                {t('auth.updatePasswordBtn')}
              </Button>
            </form>
          )}
        </div>
      </motion.div>
    </div>
  );
}
