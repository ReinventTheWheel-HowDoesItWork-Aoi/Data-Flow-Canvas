/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Building2, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAuthStore } from '@/stores/authStore';

export function ProfileCompletionModal() {
  const { t } = useTranslation();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [company, setCompany] = useState('');
  const [formError, setFormError] = useState<string | null>(null);

  const { completeProfile, isLoading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    clearError();

    // Validation
    if (!firstName.trim()) {
      setFormError(t('auth.errors.firstNameRequired'));
      return;
    }

    if (!lastName.trim()) {
      setFormError(t('auth.errors.lastNameRequired'));
      return;
    }

    const result = await completeProfile(firstName.trim(), lastName.trim(), company.trim() || undefined);
    if (!result.success) {
      setFormError(result.error || t('auth.errors.profileUpdateFailed'));
    }
  };

  const displayError = formError || error;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Transparent backdrop */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />

      {/* Modal Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="relative z-10 w-full max-w-md mx-4"
      >
        <div className="bg-bg-secondary rounded-2xl border border-border-default shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="px-8 pt-8 pb-6 text-center border-b border-border-default bg-gradient-to-b from-electric-indigo/5 to-transparent">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-14 h-14 rounded-xl mx-auto mb-4 shadow-glow" />
            <h1 className="text-2xl font-bold text-text-primary">
              {t('auth.completeProfile')}
            </h1>
            <p className="text-text-secondary mt-2">
              {t('auth.completeProfileSubtitle')}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-8 space-y-4">
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
              {t('auth.continueBtn')}
            </Button>
          </form>
        </div>
      </motion.div>
    </div>
  );
}
