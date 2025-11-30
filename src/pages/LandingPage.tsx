/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion, useInView, useScroll, useTransform } from 'framer-motion';
import {
  ArrowRight,
  Sparkles,
  Lock,
  Zap,
  Users,
  Code2,
  Github,
  Play,
  Database,
  GitBranch,
  BarChart3,
  MousePointerClick,
  Shield,
  Sun,
  Moon,
  BookOpen,
  Layers,
  Cpu,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';

// Animated dot grid background
const DotGrid = ({ isDark }: { isDark: boolean }) => (
  <div className="absolute inset-0 overflow-hidden">
    <div
      className={cn(
        'absolute inset-0',
        isDark
          ? '[background-image:radial-gradient(circle,rgba(99,102,241,0.15)_1px,transparent_1px)]'
          : '[background-image:radial-gradient(circle,rgba(99,102,241,0.08)_1px,transparent_1px)]'
      )}
      style={{ backgroundSize: '32px 32px' }}
    />
    <motion.div
      className="absolute inset-0"
      animate={{
        background: isDark
          ? [
              'radial-gradient(ellipse 80% 50% at 50% 50%, rgba(99,102,241,0.15), transparent)',
              'radial-gradient(ellipse 60% 40% at 60% 60%, rgba(139,92,246,0.12), transparent)',
              'radial-gradient(ellipse 80% 50% at 40% 40%, rgba(99,102,241,0.15), transparent)',
            ]
          : [
              'radial-gradient(ellipse 80% 50% at 50% 50%, rgba(99,102,241,0.08), transparent)',
              'radial-gradient(ellipse 60% 40% at 60% 60%, rgba(139,92,246,0.06), transparent)',
              'radial-gradient(ellipse 80% 50% at 40% 40%, rgba(99,102,241,0.08), transparent)',
            ],
      }}
      transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
    />
  </div>
);

// Enhanced floating orbs with more sophistication
const FloatingOrb = ({
  className,
  delay = 0,
  duration = 20,
  children,
}: {
  className?: string;
  delay?: number;
  duration?: number;
  children?: React.ReactNode;
}) => (
  <motion.div
    className={cn('absolute rounded-full', className)}
    animate={{
      y: [0, -40, 0],
      x: [0, 20, 0],
      scale: [1, 1.15, 1],
      rotate: [0, 5, 0],
    }}
    transition={{
      duration,
      repeat: Infinity,
      delay,
      ease: 'easeInOut',
    }}
  >
    {children}
  </motion.div>
);

// Glowing ring effect
const GlowRing = ({ delay = 0, className }: { delay?: number; className?: string }) => (
  <motion.div
    className={cn('absolute rounded-full border-2', className)}
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{
      scale: [1, 1.5, 2],
      opacity: [0.6, 0.3, 0],
    }}
    transition={{
      duration: 3,
      repeat: Infinity,
      delay,
      ease: 'easeOut',
    }}
  />
);

// Shimmer effect for badges
const ShimmerBadge = ({
  children,
  className,
  isDark,
}: {
  children: React.ReactNode;
  className?: string;
  isDark: boolean;
}) => (
  <motion.div
    className={cn('relative overflow-hidden', className)}
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay: 0.2 }}
  >
    <div
      className={cn(
        'inline-flex items-center gap-2 px-5 py-2.5 rounded-full border',
        isDark
          ? 'bg-gradient-to-r from-electric-indigo/20 via-soft-violet/20 to-electric-indigo/20 border-electric-indigo/40 backdrop-blur-xl'
          : 'bg-gradient-to-r from-white/90 via-white/80 to-white/90 border-electric-indigo/20 shadow-lg shadow-electric-indigo/10'
      )}
    >
      {children}
    </div>
    <motion.div
      className="absolute inset-0 -translate-x-full"
      animate={{ translateX: ['0%', '200%'] }}
      transition={{ duration: 2, repeat: Infinity, repeatDelay: 3, ease: 'easeInOut' }}
    >
      <div
        className={cn(
          'h-full w-1/2',
          isDark
            ? 'bg-gradient-to-r from-transparent via-white/10 to-transparent'
            : 'bg-gradient-to-r from-transparent via-white/60 to-transparent'
        )}
      />
    </motion.div>
  </motion.div>
);

// Animated node for the visual preview - enhanced
const AnimatedNode = ({
  x,
  y,
  label,
  color,
  delay = 0,
  icon: Icon,
  isDark,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
  delay?: number;
  icon: React.ElementType;
  isDark: boolean;
}) => (
  <motion.div
    className="absolute z-10"
    style={{ left: `${x}%`, top: `${y}%` }}
    initial={{ opacity: 0, scale: 0.5, y: 20 }}
    animate={{ opacity: 1, scale: 1, y: 0 }}
    transition={{ duration: 0.6, delay, type: 'spring', stiffness: 100 }}
  >
    <motion.div
      className={cn(
        'px-4 py-3 rounded-xl border-2 backdrop-blur-md relative',
        isDark
          ? 'bg-slate-800/90 border-slate-600/50 shadow-xl shadow-black/30'
          : 'bg-white/95 border-slate-200/80 shadow-xl shadow-slate-200/50'
      )}
      whileHover={{ scale: 1.08, y: -3 }}
      animate={{ y: [0, -6, 0] }}
      transition={{ duration: 4, repeat: Infinity, delay: delay * 0.5, ease: 'easeInOut' }}
    >
      {/* Glow effect behind node */}
      <div
        className={cn('absolute -inset-1 rounded-xl blur-lg opacity-50', color)}
        style={{ zIndex: -1 }}
      />
      <div className="flex items-center gap-3">
        <div
          className={cn(
            'w-9 h-9 rounded-lg flex items-center justify-center shadow-lg',
            color
          )}
        >
          <Icon size={18} className="text-white" />
        </div>
        <span
          className={cn(
            'text-sm font-semibold tracking-tight',
            isDark ? 'text-white' : 'text-slate-800'
          )}
        >
          {label}
        </span>
      </div>
      {/* Pulse ring */}
      <motion.div
        className={cn('absolute -inset-0.5 rounded-xl border-2', color.replace('bg-', 'border-'))}
        animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0, 0.5] }}
        transition={{ duration: 2, repeat: Infinity, delay }}
      />
    </motion.div>
  </motion.div>
);

// Animated connection line with flowing particles
const ConnectionLine = ({
  x1,
  y1,
  x2,
  y2,
  delay = 0,
  isDark,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  delay?: number;
  isDark: boolean;
}) => {
  const pathId = `path-${x1}-${y1}-${x2}-${y2}`;
  const gradientId = `gradient-${x1}-${y1}`;

  return (
    <motion.svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay }}
    >
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#14b8a6" />
          <stop offset="50%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* Background glow line */}
      <motion.path
        d={`M ${x1} ${y1} Q ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${(y1 + y2) / 2} T ${x2} ${y2}`}
        fill="none"
        stroke={isDark ? 'rgba(99,102,241,0.3)' : 'rgba(99,102,241,0.15)'}
        strokeWidth="8"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2, delay }}
      />
      {/* Main line */}
      <motion.path
        id={pathId}
        d={`M ${x1} ${y1} Q ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${(y1 + y2) / 2} T ${x2} ${y2}`}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth="3"
        strokeLinecap="round"
        filter="url(#glow)"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2, delay }}
      />
      {/* Animated particle */}
      <motion.circle
        r="4"
        fill="#6366f1"
        filter="url(#glow)"
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 1, 0] }}
        transition={{ duration: 2, repeat: Infinity, delay: delay + 0.5 }}
      >
        <animateMotion dur="2s" repeatCount="indefinite" begin={`${delay + 0.5}s`}>
          <mpath href={`#${pathId}`} />
        </animateMotion>
      </motion.circle>
      <motion.circle
        r="6"
        fill="#14b8a6"
        opacity="0.5"
        filter="url(#glow)"
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.5, 0.5, 0] }}
        transition={{ duration: 2, repeat: Infinity, delay: delay + 1.5 }}
      >
        <animateMotion dur="2s" repeatCount="indefinite" begin={`${delay + 1.5}s`}>
          <mpath href={`#${pathId}`} />
        </animateMotion>
      </motion.circle>
    </motion.svg>
  );
};

// Feature card with glass morphism and animated border
const FeatureCard = ({
  feature,
  index,
  isDark,
  isLarge = false,
  t,
}: {
  feature: (typeof features)[0];
  index: number;
  isDark: boolean;
  isLarge?: boolean;
  t: (key: string) => string;
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1 }}
      className={cn('group relative', isLarge && 'md:col-span-2')}
    >
      {/* Animated gradient border */}
      <div className="absolute -inset-[1px] rounded-2xl overflow-hidden">
        <motion.div
          className={cn(
            'absolute inset-0',
            isDark
              ? 'bg-gradient-to-r from-electric-indigo/50 via-soft-violet/50 to-fresh-teal/50'
              : 'bg-gradient-to-r from-electric-indigo/30 via-soft-violet/30 to-fresh-teal/30'
          )}
          animate={{
            background: isDark
              ? [
                  'linear-gradient(0deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5), rgba(20,184,166,0.5))',
                  'linear-gradient(90deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5), rgba(20,184,166,0.5))',
                  'linear-gradient(180deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5), rgba(20,184,166,0.5))',
                  'linear-gradient(270deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5), rgba(20,184,166,0.5))',
                  'linear-gradient(360deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5), rgba(20,184,166,0.5))',
                ]
              : [
                  'linear-gradient(0deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3), rgba(20,184,166,0.3))',
                  'linear-gradient(90deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3), rgba(20,184,166,0.3))',
                  'linear-gradient(180deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3), rgba(20,184,166,0.3))',
                  'linear-gradient(270deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3), rgba(20,184,166,0.3))',
                  'linear-gradient(360deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3), rgba(20,184,166,0.3))',
                ],
          }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          style={{ opacity: 0 }}
        />
        <motion.div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: isDark
              ? 'linear-gradient(135deg, rgba(99,102,241,0.6), rgba(139,92,246,0.4), rgba(20,184,166,0.6))'
              : 'linear-gradient(135deg, rgba(99,102,241,0.4), rgba(139,92,246,0.3), rgba(20,184,166,0.4))',
          }}
        />
      </div>

      <motion.div
        className={cn(
          'relative p-6 md:p-8 rounded-2xl h-full transition-all duration-500',
          isDark
            ? 'bg-slate-900/80 backdrop-blur-xl'
            : 'bg-white/80 backdrop-blur-xl shadow-lg',
          isLarge && 'flex flex-col md:flex-row md:items-center gap-6'
        )}
        whileHover={{ y: -5 }}
      >
        {/* Icon with glow */}
        <div className="relative">
          <motion.div
            className={cn(
              'w-14 h-14 rounded-xl flex items-center justify-center relative z-10',
              'transition-transform duration-300 group-hover:scale-110',
              feature.bgColor
            )}
            whileHover={{ rotate: [0, -10, 10, 0] }}
            transition={{ duration: 0.5 }}
          >
            <feature.icon size={28} className={feature.iconColor} />
          </motion.div>
          {/* Glow behind icon */}
          <div
            className={cn(
              'absolute inset-0 rounded-xl blur-xl opacity-0 group-hover:opacity-60 transition-opacity duration-500',
              feature.bgColor
            )}
          />
        </div>

        <div className={isLarge ? 'flex-1' : 'mt-5'}>
          <h3
            className={cn(
              'text-xl font-bold mb-2 tracking-tight',
              isDark ? 'text-white' : 'text-slate-900'
            )}
          >
            {t(feature.titleKey)}
          </h3>
          <p
            className={cn(
              'text-sm leading-relaxed',
              isDark ? 'text-slate-400' : 'text-slate-600'
            )}
          >
            {t(feature.descriptionKey)}
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
};

// Step card with animated number
const StepCard = ({
  step,
  index,
  isDark,
  total,
  t,
}: {
  step: (typeof steps)[0];
  index: number;
  isDark: boolean;
  total: number;
  t: (key: string) => string;
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <motion.div
      ref={ref}
      className="relative text-center"
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.2 }}
    >
      {/* Connecting line */}
      {index < total - 1 && (
        <div
          className={cn(
            'hidden md:block absolute top-8 left-[60%] w-[80%] h-[2px]',
            isDark ? 'bg-slate-700' : 'bg-slate-200'
          )}
        >
          <motion.div
            className="h-full bg-gradient-to-r from-electric-indigo to-soft-violet"
            initial={{ scaleX: 0 }}
            animate={isInView ? { scaleX: 1 } : {}}
            transition={{ duration: 0.8, delay: index * 0.2 + 0.3 }}
            style={{ transformOrigin: 'left' }}
          />
        </div>
      )}

      {/* Number badge */}
      <motion.div
        className="relative inline-flex"
        whileHover={{ scale: 1.1 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-electric-indigo to-soft-violet flex items-center justify-center text-2xl font-bold text-white shadow-glow relative overflow-hidden">
          <span className="relative z-10">{index + 1}</span>
          {/* Shine effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full"
            animate={{ translateX: ['100%', '-100%'] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 2 }}
          />
        </div>
        {/* Pulse rings */}
        <GlowRing
          className="border-electric-indigo/50 w-16 h-16 top-0 left-0"
          delay={index * 0.3}
        />
      </motion.div>

      <h3
        className={cn(
          'text-xl font-bold mt-6 mb-3 tracking-tight',
          isDark ? 'text-white' : 'text-slate-900'
        )}
      >
        {t(step.titleKey)}
      </h3>
      <p className={cn('max-w-xs mx-auto', isDark ? 'text-slate-400' : 'text-slate-600')}>
        {t(step.descriptionKey)}
      </p>
    </motion.div>
  );
};

export default function LandingPage() {
  const { t } = useTranslation();
  const { isDarkMode, setDarkMode, toggleDarkMode } = useUIStore();
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });

  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95]);

  // Set light mode as default on landing page mount
  useEffect(() => {
    setDarkMode(false);
  }, [setDarkMode]);

  return (
    <div
      className={cn(
        'min-h-screen overflow-hidden transition-colors duration-500',
        isDarkMode
          ? 'bg-[#0a0a14] text-white'
          : 'bg-gradient-to-b from-slate-50 via-white to-slate-50 text-slate-900'
      )}
    >
      {/* Sophisticated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <DotGrid isDark={isDarkMode} />

        {/* Enhanced floating orbs */}
        <FloatingOrb
          className={cn(
            'w-[700px] h-[700px] -top-48 -left-48 blur-3xl',
            isDarkMode ? 'bg-electric-indigo/30' : 'bg-electric-indigo/15'
          )}
          delay={0}
          duration={25}
        />
        <FloatingOrb
          className={cn(
            'w-[600px] h-[600px] top-1/4 -right-32 blur-3xl',
            isDarkMode ? 'bg-soft-violet/25' : 'bg-soft-violet/12'
          )}
          delay={3}
          duration={22}
        />
        <FloatingOrb
          className={cn(
            'w-[500px] h-[500px] bottom-0 left-1/4 blur-3xl',
            isDarkMode ? 'bg-fresh-teal/15' : 'bg-fresh-teal/8'
          )}
          delay={5}
          duration={28}
        />

        {/* Noise texture overlay */}
        <div
          className={cn(
            'absolute inset-0 opacity-[0.015]',
            isDarkMode ? 'opacity-[0.03]' : ''
          )}
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          }}
        />

        {/* Gradient overlay for dark mode */}
        {isDarkMode && (
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/0 via-[#0a0a14] to-[#0a0a14]" />
        )}
      </div>

      {/* Header */}
      <header className="relative container mx-auto px-4 sm:px-6 py-4 sm:py-6 flex items-center justify-between">
        <motion.div
          className="flex items-center gap-2 sm:gap-3 flex-shrink-0"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        >
          <motion.div
            className="relative"
            whileHover={{ scale: 1.05 }}
            transition={{ type: 'spring', stiffness: 400 }}
          >
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-9 h-9 sm:w-11 sm:h-11 rounded-xl shadow-glow" />
            <div className="absolute -inset-1 rounded-xl bg-gradient-to-br from-electric-indigo to-soft-violet blur-lg opacity-40" />
          </motion.div>
          <span className="text-base sm:text-xl font-bold tracking-tight hidden xs:block">Data Flow Canvas</span>
        </motion.div>

        <motion.nav
          className="flex items-center gap-1.5 sm:gap-3"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        >
          {/* Dark/Light Mode Toggle */}
          <motion.button
            onClick={toggleDarkMode}
            className={cn(
              'p-2 sm:p-2.5 rounded-xl transition-all duration-300',
              isDarkMode
                ? 'text-slate-300 hover:text-white hover:bg-slate-800/80 backdrop-blur-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-white/80 hover:shadow-md backdrop-blur-sm'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <motion.div
              initial={false}
              animate={{ rotate: isDarkMode ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              {isDarkMode ? <Sun size={18} className="sm:w-5 sm:h-5" /> : <Moon size={18} className="sm:w-5 sm:h-5" />}
            </motion.div>
          </motion.button>

          <motion.a
            href="https://github.com/ReinventTheWheel-HowDoesItWork-Aoi/Data-Flow-Canvas"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'hidden sm:flex items-center gap-2 px-3 py-2 rounded-xl transition-all duration-300',
              isDarkMode
                ? 'text-slate-300 hover:text-white hover:bg-slate-800/80'
                : 'text-slate-600 hover:text-slate-900 hover:bg-white/80 hover:shadow-md'
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Github size={20} />
            <span className="hidden md:inline font-medium">GitHub</span>
          </motion.a>

          {/* Language selector - icon on mobile, full on desktop */}
          <div className="hidden sm:block">
            <LanguageSelector variant="full" />
          </div>
          <div className="block sm:hidden">
            <LanguageSelector variant="icon" />
          </div>

          <Link to="/editor">
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button variant="primary" size="sm" className="shadow-glow text-xs sm:text-sm px-2.5 sm:px-4">
                <Play size={14} className="sm:w-4 sm:h-4 mr-1 sm:mr-1.5" />
                <span className="hidden xs:inline">{t('header.launchApp')}</span>
                <span className="xs:hidden">Launch</span>
              </Button>
            </motion.div>
          </Link>
        </motion.nav>
      </header>

      {/* Hero Section */}
      <section ref={heroRef} className="relative container mx-auto px-4 sm:px-6 pt-12 sm:pt-20 pb-16 sm:pb-32">
        <motion.div
          style={{ opacity: heroOpacity, scale: heroScale }}
          className="grid lg:grid-cols-2 gap-16 items-center"
        >
          {/* Left Column - Text Content */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          >
            <ShimmerBadge isDark={isDarkMode}>
              <Sparkles size={16} className="text-electric-indigo" />
              <span
                className={cn(
                  'text-sm font-semibold',
                  isDarkMode ? 'text-slate-200' : 'text-slate-700'
                )}
              >
                {t('landing.badge.browserBased')}
              </span>
              <span className={isDarkMode ? 'text-slate-600' : 'text-slate-300'}>•</span>
              <Shield size={14} className="text-fresh-teal" />
              <span
                className={cn(
                  'text-sm font-semibold',
                  isDarkMode ? 'text-slate-200' : 'text-slate-700'
                )}
              >
                {t('landing.badge.dataLocal')}
              </span>
            </ShimmerBadge>

            <motion.h1
              className="text-3xl xs:text-4xl sm:text-5xl lg:text-7xl font-bold leading-[1.1] mt-6 sm:mt-8 mb-4 sm:mb-6 tracking-tight"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <span className="gradient-text inline-block">{t('landing.hero.titleLine1')}</span>
              <br />
              <span className={cn('inline-block', isDarkMode ? 'text-white' : 'text-slate-900')}>
                {t('landing.hero.titleLine2')}
              </span>
            </motion.h1>

            <motion.p
              className={cn(
                'text-base sm:text-xl lg:text-2xl mb-6 sm:mb-8 leading-relaxed max-w-xl',
                isDarkMode ? 'text-slate-300' : 'text-slate-600'
              )}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              {t('landing.hero.description')}
            </motion.p>

            {/* Tech Stack Badges */}
            <motion.div
              className="flex flex-wrap gap-2 mb-10"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              {['Python', 'pandas', 'NumPy', 'scikit-learn', 'WebAssembly'].map((tech, i) => (
                <motion.span
                  key={tech}
                  className={cn(
                    'px-4 py-1.5 text-xs font-semibold rounded-full border transition-all duration-300',
                    isDarkMode
                      ? 'bg-slate-800/80 border-slate-700/60 text-slate-300 hover:border-electric-indigo/50 hover:bg-slate-800'
                      : 'bg-white/90 border-slate-200 text-slate-700 shadow-sm hover:border-electric-indigo/50 hover:shadow-md'
                  )}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: 0.5 + i * 0.08 }}
                  whileHover={{ y: -2 }}
                >
                  {tech}
                </motion.span>
              ))}
            </motion.div>

            <motion.div
              className="flex flex-wrap items-center gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <Link to="/editor">
                <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                  <Button
                    variant="primary"
                    size="lg"
                    rightIcon={<ArrowRight size={20} />}
                    className="shadow-glow text-base font-semibold px-8"
                  >
                    {t('landing.hero.ctaPrimary')}
                  </Button>
                </motion.div>
              </Link>
              <Link to="/projects">
                <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                  <Button variant="secondary" size="lg" className="text-base font-semibold">
                    <MousePointerClick size={18} className="mr-2" />
                    {t('landing.hero.ctaExamples')}
                  </Button>
                </motion.div>
              </Link>
              <Link to="/help">
                <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                  <Button variant="ghost" size="lg" className="text-base font-semibold">
                    <BookOpen size={18} className="mr-2" />
                    {t('landing.hero.ctaGuide')}
                  </Button>
                </motion.div>
              </Link>
            </motion.div>

            {/* Trust Indicator */}
            <motion.p
              className={cn(
                'mt-8 text-sm flex items-center gap-2',
                isDarkMode ? 'text-slate-500' : 'text-slate-500'
              )}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <Lock size={14} />
              {t('landing.hero.trustText')}
            </motion.p>
          </motion.div>

          {/* Right Column - Visual Preview */}
          <motion.div
            className="relative h-[450px] lg:h-[550px] hidden lg:block"
            initial={{ opacity: 0, scale: 0.85, x: 40 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ duration: 1, delay: 0.3, ease: 'easeOut' }}
          >
            {/* Multi-layer glow effect */}
            <div
              className={cn(
                'absolute inset-0 rounded-3xl blur-3xl',
                isDarkMode
                  ? 'bg-gradient-to-r from-electric-indigo/25 via-soft-violet/20 to-fresh-teal/25'
                  : 'bg-gradient-to-r from-electric-indigo/15 via-soft-violet/10 to-fresh-teal/15'
              )}
            />
            <div
              className={cn(
                'absolute inset-4 rounded-2xl blur-2xl',
                isDarkMode
                  ? 'bg-gradient-to-br from-electric-indigo/20 to-soft-violet/20'
                  : 'bg-gradient-to-br from-electric-indigo/10 to-soft-violet/10'
              )}
            />

            {/* Preview Container */}
            <motion.div
              className={cn(
                'relative h-full rounded-2xl border-2 backdrop-blur-md overflow-hidden',
                isDarkMode
                  ? 'border-slate-700/60 bg-slate-900/70'
                  : 'border-slate-200/80 bg-white/80 shadow-2xl'
              )}
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
            >
              {/* Animated border gradient */}
              <div className="absolute inset-0 rounded-2xl overflow-hidden">
                <motion.div
                  className="absolute -inset-[100%] opacity-30"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
                  style={{
                    background:
                      'conic-gradient(from 0deg, #6366f1, #8b5cf6, #14b8a6, #6366f1)',
                  }}
                />
              </div>

              {/* Mock Editor Header */}
              <div
                className={cn(
                  'relative flex items-center gap-3 px-5 py-4 border-b',
                  isDarkMode
                    ? 'border-slate-700/60 bg-slate-800/60'
                    : 'border-slate-200/80 bg-slate-50/90'
                )}
              >
                <div className="flex gap-2">
                  <motion.div
                    className="w-3.5 h-3.5 rounded-full bg-warm-coral shadow-sm"
                    whileHover={{ scale: 1.2 }}
                  />
                  <motion.div
                    className="w-3.5 h-3.5 rounded-full bg-golden-amber shadow-sm"
                    whileHover={{ scale: 1.2 }}
                  />
                  <motion.div
                    className="w-3.5 h-3.5 rounded-full bg-fresh-teal shadow-sm"
                    whileHover={{ scale: 1.2 }}
                  />
                </div>
                <div className="flex items-center gap-2 ml-2">
                  <Layers
                    size={14}
                    className={isDarkMode ? 'text-slate-500' : 'text-slate-400'}
                  />
                  <span
                    className={cn(
                      'text-xs font-medium',
                      isDarkMode ? 'text-slate-500' : 'text-slate-400'
                    )}
                  >
                    Data Flow Canvas
                  </span>
                </div>
              </div>

              {/* Animated Nodes with data flow */}
              <div className="relative h-full p-4">
                <ConnectionLine
                  x1={18}
                  y1={25}
                  x2={42}
                  y2={48}
                  delay={0.8}
                  isDark={isDarkMode}
                />
                <ConnectionLine
                  x1={52}
                  y1={48}
                  x2={78}
                  y2={35}
                  delay={1.2}
                  isDark={isDarkMode}
                />
                <ConnectionLine
                  x1={52}
                  y1={55}
                  x2={78}
                  y2={72}
                  delay={1.5}
                  isDark={isDarkMode}
                />

                <AnimatedNode
                  x={3}
                  y={15}
                  label={t('landing.previewNodes.loadCsv')}
                  color="bg-electric-indigo"
                  icon={Database}
                  delay={0.5}
                  isDark={isDarkMode}
                />
                <AnimatedNode
                  x={32}
                  y={40}
                  label={t('landing.previewNodes.filterRows')}
                  color="bg-soft-violet"
                  icon={GitBranch}
                  delay={0.7}
                  isDark={isDarkMode}
                />
                <AnimatedNode
                  x={65}
                  y={25}
                  label={t('landing.previewNodes.groupBy')}
                  color="bg-fresh-teal"
                  icon={Cpu}
                  delay={0.9}
                  isDark={isDarkMode}
                />
                <AnimatedNode
                  x={65}
                  y={62}
                  label={t('landing.previewNodes.barChart')}
                  color="bg-golden-amber"
                  icon={BarChart3}
                  delay={1.1}
                  isDark={isDarkMode}
                />
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section - Bento Grid Style */}
      <section className="relative container mx-auto px-4 sm:px-6 py-16 sm:py-32">
        <motion.div
          className="text-center mb-10 sm:mb-20"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <motion.h2
            className="text-2xl sm:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6 tracking-tight"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {t('landing.features.sectionTitle')}
            <span className="gradient-text"> {t('landing.features.sectionTitleHighlight')}</span>
          </motion.h2>
          <motion.p
            className={cn(
              'text-base sm:text-lg lg:text-xl max-w-2xl mx-auto',
              isDarkMode ? 'text-slate-400' : 'text-slate-600'
            )}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {t('landing.features.sectionDescription')}
          </motion.p>
        </motion.div>

        {/* Bento Grid Layout */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.titleKey}
              feature={feature}
              index={index}
              isDark={isDarkMode}
              isLarge={index === 0 || index === 3}
              t={t}
            />
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative container mx-auto px-4 sm:px-6 py-16 sm:py-32">
        <motion.div
          className="text-center mb-10 sm:mb-20"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <h2 className="text-2xl sm:text-4xl lg:text-5xl font-bold mb-4 tracking-tight">
            {t('landing.howItWorks.title')} <span className="gradient-text">{t('landing.howItWorks.titleHighlight')}</span>
          </h2>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 sm:gap-12 max-w-5xl mx-auto">
          {steps.map((step, index) => (
            <StepCard
              key={step.titleKey}
              step={step}
              index={index}
              isDark={isDarkMode}
              total={steps.length}
              t={t}
            />
          ))}
        </div>
      </section>

      {/* CTA Section with animated gradient border */}
      <section className="relative container mx-auto px-4 sm:px-6 py-16 sm:py-32">
        <motion.div
          className="relative rounded-3xl overflow-hidden"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          {/* Animated gradient border */}
          <div className="absolute -inset-[2px] rounded-3xl overflow-hidden">
            <motion.div
              className="absolute inset-0"
              animate={{ rotate: 360 }}
              transition={{ duration: 6, repeat: Infinity, ease: 'linear' }}
              style={{
                background: isDarkMode
                  ? 'conic-gradient(from 0deg, #6366f1, #8b5cf6, #14b8a6, #f59e0b, #6366f1)'
                  : 'conic-gradient(from 0deg, #6366f1, #8b5cf6, #14b8a6, #f59e0b, #6366f1)',
              }}
            />
          </div>

          <div
            className={cn(
              'relative p-8 sm:p-16 text-center rounded-3xl',
              isDarkMode
                ? 'bg-gradient-to-br from-slate-900 via-slate-900/95 to-slate-800'
                : 'bg-gradient-to-br from-white via-white to-slate-50'
            )}
          >
            {/* Background glow */}
            <div
              className={cn(
                'absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] rounded-full blur-3xl',
                isDarkMode ? 'bg-electric-indigo/15' : 'bg-electric-indigo/8'
              )}
            />

            <div className="relative">
              <motion.h2
                className={cn(
                  'text-2xl sm:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6 tracking-tight',
                  isDarkMode ? 'text-white' : 'text-slate-900'
                )}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                {t('landing.cta.title')}
              </motion.h2>
              <motion.p
                className={cn(
                  'text-base sm:text-lg lg:text-xl max-w-xl mx-auto mb-8 sm:mb-10',
                  isDarkMode ? 'text-slate-300' : 'text-slate-600'
                )}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                {t('landing.cta.description')}
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <Link to="/editor">
                  <motion.div
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    className="inline-block"
                  >
                    <Button
                      variant="primary"
                      size="lg"
                      rightIcon={<ArrowRight size={20} />}
                      className="shadow-glow text-base font-semibold px-10 py-4"
                    >
                      {t('landing.cta.button')}
                    </Button>
                  </motion.div>
                </Link>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer
        className={cn(
          'relative container mx-auto px-4 sm:px-6 py-8 sm:py-12 border-t',
          isDarkMode ? 'border-slate-800/60' : 'border-slate-200'
        )}
      >
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 sm:gap-6">
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-9 h-9 rounded-lg shadow-sm" />
            <span className={cn('font-medium', isDarkMode ? 'text-slate-400' : 'text-slate-500')}>
              {t('landing.footer.openSource')}
            </span>
          </motion.div>
          <motion.div
            className={cn('flex flex-wrap items-center justify-center gap-4 sm:gap-8', isDarkMode ? 'text-slate-400' : 'text-slate-500')}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <a
              href="https://github.com/ReinventTheWheel-HowDoesItWork-Aoi/Data-Flow-Canvas/blob/main/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                'text-sm font-medium transition-colors hover:underline underline-offset-4',
                isDarkMode ? 'hover:text-white' : 'hover:text-slate-900'
              )}
            >
              {t('landing.footer.documentation')}
            </a>
            <Link
              to="/terms"
              className={cn(
                'text-sm font-medium transition-colors hover:underline underline-offset-4',
                isDarkMode ? 'hover:text-white' : 'hover:text-slate-900'
              )}
            >
              {t('landing.footer.terms')}
            </Link>
            <Link
              to="/privacy"
              className={cn(
                'text-sm font-medium transition-colors hover:underline underline-offset-4',
                isDarkMode ? 'hover:text-white' : 'hover:text-slate-900'
              )}
            >
              {t('landing.footer.privacy')}
            </Link>
            <a
              href="https://github.com/ReinventTheWheel-HowDoesItWork-Aoi/Data-Flow-Canvas"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                'text-sm font-medium flex items-center gap-2 transition-colors hover:underline underline-offset-4',
                isDarkMode ? 'hover:text-white' : 'hover:text-slate-900'
              )}
            >
              <Github size={16} />
              GitHub
            </a>
          </motion.div>
        </div>
      </footer>
    </div>
  );
}

const features = [
  {
    titleKey: 'landing.features.privacyFirst.title',
    descriptionKey: 'landing.features.privacyFirst.description',
    icon: Lock,
    bgColor: 'bg-fresh-teal/20',
    iconColor: 'text-fresh-teal',
  },
  {
    titleKey: 'landing.features.quickStart.title',
    descriptionKey: 'landing.features.quickStart.description',
    icon: Zap,
    bgColor: 'bg-electric-indigo/20',
    iconColor: 'text-electric-indigo',
  },
  {
    titleKey: 'landing.features.collaborate.title',
    descriptionKey: 'landing.features.collaborate.description',
    icon: Users,
    bgColor: 'bg-soft-violet/20',
    iconColor: 'text-soft-violet',
  },
  {
    titleKey: 'landing.features.pythonPowered.title',
    descriptionKey: 'landing.features.pythonPowered.description',
    icon: Code2,
    bgColor: 'bg-golden-amber/20',
    iconColor: 'text-golden-amber',
  },
];

const steps = [
  {
    titleKey: 'landing.howItWorks.step1.title',
    descriptionKey: 'landing.howItWorks.step1.description',
  },
  {
    titleKey: 'landing.howItWorks.step2.title',
    descriptionKey: 'landing.howItWorks.step2.description',
  },
  {
    titleKey: 'landing.howItWorks.step3.title',
    descriptionKey: 'landing.howItWorks.step3.description',
  },
];
