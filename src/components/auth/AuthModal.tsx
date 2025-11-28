/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, User, Building2, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { useAuthStore } from '@/stores/authStore';
import { cn } from '@/lib/utils/cn';

type AuthMode = 'signin' | 'signup' | 'forgot';

export function AuthModal() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<AuthMode>('signup');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [company, setCompany] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [resetEmailSent, setResetEmailSent] = useState(false);

  const { signIn, signUp, resetPassword, isLoading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    clearError();

    // Validation
    if (!email.trim()) {
      setFormError(t('auth.errors.emailRequired'));
      return;
    }

    // Handle forgot password
    if (mode === 'forgot') {
      const result = await resetPassword(email);
      if (result.success) {
        setResetEmailSent(true);
      } else {
        setFormError(result.error || t('auth.errors.resetFailed'));
      }
      return;
    }

    if (!password.trim()) {
      setFormError(t('auth.errors.passwordRequired'));
      return;
    }
    if (password.length < 6) {
      setFormError(t('auth.errors.passwordLength'));
      return;
    }

    if (mode === 'signup') {
      if (!firstName.trim()) {
        setFormError(t('auth.errors.firstNameRequired'));
        return;
      }
      if (!lastName.trim()) {
        setFormError(t('auth.errors.lastNameRequired'));
        return;
      }

      const result = await signUp(email, password, firstName, lastName, company || undefined);
      if (!result.success) {
        setFormError(result.error || t('auth.errors.signUpFailed'));
      }
    } else {
      const result = await signIn(email, password);
      if (!result.success) {
        setFormError(result.error || t('auth.errors.signInFailed'));
      }
    }
  };

  const toggleMode = () => {
    setMode(mode === 'signin' ? 'signup' : 'signin');
    setFormError(null);
    setResetEmailSent(false);
    clearError();
  };

  const goToForgotPassword = () => {
    setMode('forgot');
    setFormError(null);
    setResetEmailSent(false);
    clearError();
  };

  const backToSignIn = () => {
    setMode('signin');
    setFormError(null);
    setResetEmailSent(false);
    clearError();
  };

  const displayError = formError || error;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Transparent backdrop - lets users see the editor behind */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />

      {/* Auth Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="relative z-10 w-full max-w-md mx-4"
      >
        <div className="bg-bg-secondary rounded-2xl border border-border-default shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="px-8 pt-8 pb-6 text-center border-b border-border-default bg-gradient-to-b from-electric-indigo/5 to-transparent">
            <div className="flex justify-end mb-2">
              <LanguageSelector />
            </div>
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-14 h-14 rounded-xl mx-auto mb-4 shadow-glow" />
            <h1 className="text-2xl font-bold text-text-primary">
              {mode === 'signup' ? t('auth.createAccount') : mode === 'forgot' ? t('auth.forgotPassword') : t('auth.welcomeBack')}
            </h1>
            <p className="text-text-secondary mt-2">
              {mode === 'signup'
                ? t('auth.signUpSubtitle')
                : mode === 'forgot'
                ? t('auth.forgotSubtitle')
                : t('auth.signInSubtitle')}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-8 space-y-4">
            <AnimatePresence mode="wait">
              {mode === 'signup' && (
                <motion.div
                  key="signup-fields"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="space-y-4"
                >
                  <div className="grid grid-cols-2 gap-4">
                    <Input
                      label={t('auth.firstName')}
                      placeholder="John"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      leftIcon={<User size={18} />}
                      disabled={isLoading}
                    />
                    <Input
                      label={t('auth.lastName')}
                      placeholder="Doe"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      disabled={isLoading}
                    />
                  </div>
                  <Input
                    label={t('auth.company')}
                    placeholder="Acme Inc."
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    leftIcon={<Building2 size={18} />}
                    disabled={isLoading}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <Input
              label={t('auth.email')}
              type="email"
              placeholder="john@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              leftIcon={<Mail size={18} />}
              disabled={isLoading}
            />

            {mode !== 'forgot' && (
              <div>
                <Input
                  label={t('auth.password')}
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  leftIcon={<Lock size={18} />}
                  disabled={isLoading}
                />
                {mode === 'signin' && (
                  <button
                    type="button"
                    onClick={goToForgotPassword}
                    className="text-sm text-electric-indigo hover:text-soft-violet font-medium transition-colors mt-2"
                    disabled={isLoading}
                  >
                    {t('auth.forgotPasswordLink')}
                  </button>
                )}
              </div>
            )}

            {/* Reset Email Sent Success */}
            <AnimatePresence>
              {resetEmailSent && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-3 rounded-lg bg-fresh-mint/10 border border-fresh-mint/20"
                >
                  <p className="text-sm text-fresh-mint">{t('auth.resetEmailSent')}</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error Message */}
            <AnimatePresence>
              {displayError && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-3 rounded-lg bg-warm-coral/10 border border-warm-coral/20"
                >
                  <p className="text-sm text-warm-coral">{displayError}</p>
                </motion.div>
              )}
            </AnimatePresence>

            {!resetEmailSent && (
              <Button
                type="submit"
                variant="primary"
                size="lg"
                isLoading={isLoading}
                className="w-full mt-6"
                rightIcon={!isLoading ? <ArrowRight size={18} /> : undefined}
              >
                {mode === 'signup' ? t('auth.createAccountBtn') : mode === 'forgot' ? t('auth.sendResetLink') : t('auth.signInBtn')}
              </Button>
            )}
          </form>

          {/* Footer */}
          <div className="px-8 pb-8 text-center">
            {mode === 'forgot' ? (
              <button
                type="button"
                onClick={backToSignIn}
                className="text-sm text-electric-indigo hover:text-soft-violet font-medium transition-colors"
                disabled={isLoading}
              >
                {t('auth.backToSignIn')}
              </button>
            ) : (
              <p className="text-text-secondary text-sm">
                {mode === 'signup' ? t('auth.alreadyHaveAccount') : t('auth.dontHaveAccount')}{' '}
                <button
                  type="button"
                  onClick={toggleMode}
                  className="text-electric-indigo hover:text-soft-violet font-medium transition-colors"
                  disabled={isLoading}
                >
                  {mode === 'signup' ? t('auth.signIn') : t('auth.signUp')}
                </button>
              </p>
            )}
          </div>
        </div>

        {/* Privacy note */}
        <p className="text-center text-text-muted text-xs mt-4 px-4">
          {t('auth.privacyNote')}
        </p>
      </motion.div>
    </div>
  );
}
