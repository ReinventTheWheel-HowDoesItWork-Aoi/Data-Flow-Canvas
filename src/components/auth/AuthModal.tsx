/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, ArrowRight, CheckCircle, User, Building2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { useAuthStore } from '@/stores/authStore';
import { cn } from '@/lib/utils/cn';

type AuthMode = 'signin' | 'signup';

export function AuthModal() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<AuthMode>('signup');
  const [email, setEmail] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [company, setCompany] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [magicLinkSent, setMagicLinkSent] = useState(false);

  const { signInWithMagicLink, signUpWithMagicLink, isLoading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    clearError();

    // Email validation
    if (!email.trim()) {
      setFormError(t('auth.errors.emailRequired'));
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setFormError(t('auth.errors.emailInvalid'));
      return;
    }

    // Sign up validation
    if (mode === 'signup') {
      if (!firstName.trim()) {
        setFormError(t('auth.errors.firstNameRequired'));
        return;
      }
      if (!lastName.trim()) {
        setFormError(t('auth.errors.lastNameRequired'));
        return;
      }

      const result = await signUpWithMagicLink(email, firstName.trim(), lastName.trim(), company.trim() || undefined);
      if (result.success) {
        setMagicLinkSent(true);
      } else {
        setFormError(result.error || t('auth.errors.magicLinkFailed'));
      }
    } else {
      const result = await signInWithMagicLink(email);
      if (result.success) {
        setMagicLinkSent(true);
      } else {
        setFormError(result.error || t('auth.errors.magicLinkFailed'));
      }
    }
  };

  const handleResend = () => {
    setMagicLinkSent(false);
    setFormError(null);
    clearError();
  };

  const toggleMode = () => {
    setMode(mode === 'signin' ? 'signup' : 'signin');
    setFormError(null);
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
              {magicLinkSent
                ? t('auth.checkEmail')
                : mode === 'signup'
                  ? t('auth.createAccount')
                  : t('auth.welcomeBack')}
            </h1>
            <p className="text-text-secondary mt-2">
              {magicLinkSent
                ? t('auth.magicLinkSentSubtitle')
                : mode === 'signup'
                  ? t('auth.signUpSubtitle')
                  : t('auth.signInSubtitle')}
            </p>
          </div>

          {/* Mode Toggle */}
          {!magicLinkSent && (
            <div className="px-8 pt-6">
              <div className="flex rounded-xl bg-bg-tertiary p-1">
                <button
                  type="button"
                  onClick={() => { setMode('signup'); setFormError(null); clearError(); }}
                  className={cn(
                    'flex-1 py-2.5 text-sm font-medium rounded-lg transition-all',
                    mode === 'signup'
                      ? 'bg-bg-secondary text-text-primary shadow-sm'
                      : 'text-text-secondary hover:text-text-primary'
                  )}
                >
                  {t('auth.signUp')}
                </button>
                <button
                  type="button"
                  onClick={() => { setMode('signin'); setFormError(null); clearError(); }}
                  className={cn(
                    'flex-1 py-2.5 text-sm font-medium rounded-lg transition-all',
                    mode === 'signin'
                      ? 'bg-bg-secondary text-text-primary shadow-sm'
                      : 'text-text-secondary hover:text-text-primary'
                  )}
                >
                  {t('auth.signIn')}
                </button>
              </div>
            </div>
          )}

          {/* Form */}
          <div className="p-8">
            <AnimatePresence mode="wait">
              {magicLinkSent ? (
                <motion.div
                  key="success"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-center space-y-4"
                >
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.1, type: 'spring' }}
                    className="w-16 h-16 mx-auto rounded-full bg-fresh-mint/10 flex items-center justify-center"
                  >
                    <CheckCircle size={32} className="text-fresh-mint" />
                  </motion.div>
                  <div>
                    <p className="text-text-primary font-medium">{email}</p>
                    <p className="text-text-secondary text-sm mt-2">
                      {t('auth.magicLinkInstructions')}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleResend}
                    className="text-sm text-electric-indigo hover:text-soft-violet font-medium transition-colors"
                  >
                    {t('auth.useDifferentEmail')}
                  </button>
                </motion.div>
              ) : (
                <motion.form
                  key={mode}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  onSubmit={handleSubmit}
                  className="space-y-4"
                >
                  {/* Sign Up Fields */}
                  {mode === 'signup' && (
                    <>
                      <div className="grid grid-cols-2 gap-4">
                        <Input
                          label={t('auth.firstName')}
                          placeholder="John"
                          value={firstName}
                          onChange={(e) => setFirstName(e.target.value)}
                          leftIcon={<User size={18} />}
                          disabled={isLoading}
                          autoFocus
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
                        placeholder="Acme Inc. (optional)"
                        value={company}
                        onChange={(e) => setCompany(e.target.value)}
                        leftIcon={<Building2 size={18} />}
                        disabled={isLoading}
                      />
                    </>
                  )}

                  <Input
                    label={t('auth.email')}
                    type="email"
                    placeholder="john@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    leftIcon={<Mail size={18} />}
                    disabled={isLoading}
                    autoFocus={mode === 'signin'}
                  />

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

                  <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    isLoading={isLoading}
                    className="w-full mt-6"
                    rightIcon={!isLoading ? <ArrowRight size={18} /> : undefined}
                  >
                    {t('auth.sendMagicLink')}
                  </Button>
                </motion.form>
              )}
            </AnimatePresence>
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
