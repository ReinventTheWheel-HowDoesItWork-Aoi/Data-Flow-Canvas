/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import {
  ChevronLeft,
  Shield,
  Database,
  Lock,
  Eye,
  Server,
  Users,
  Trash2,
  Mail,
  Globe,
  FileText,
} from 'lucide-react';
import { cn } from '@/lib/utils/cn';

const GITHUB_REPO = 'https://github.com/ReinventTheWheel-HowDoesItWork-Aoi/Data-Flow-Canvas';
const LAST_UPDATED = 'November 27, 2025';

interface SectionProps {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
  color: string;
}

const Section = ({ icon: Icon, title, children, color }: SectionProps) => (
  <section className="mb-10">
    <div className="flex items-center gap-3 mb-4">
      <div
        className={cn('p-2 rounded-lg')}
        style={{ backgroundColor: `${color}15` }}
      >
        <Icon size={20} style={{ color }} />
      </div>
      <h2 className="text-h2 text-text-primary font-semibold">{title}</h2>
    </div>
    <div className="text-text-secondary leading-relaxed space-y-4 pl-11">
      {children}
    </div>
  </section>
);

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-bg-secondary/80 backdrop-blur-xl border-b border-border-default">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link
            to="/"
            className="group flex items-center gap-2 text-text-muted hover:text-text-primary transition-all duration-200"
          >
            <div className="p-1.5 rounded-lg bg-bg-tertiary group-hover:bg-electric-indigo/10 transition-colors">
              <ChevronLeft size={18} className="group-hover:text-electric-indigo transition-colors" />
            </div>
            <span className="text-small font-medium">Back to Home</span>
          </Link>
          <div className="flex-1" />
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-8 h-8 rounded-lg shadow-glow" />
            <span className="hidden sm:block font-semibold text-text-primary">Data Flow Canvas</span>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-border-default">
        <div className="absolute inset-0 bg-gradient-to-br from-fresh-teal/5 via-transparent to-electric-indigo/5" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-fresh-teal/10 to-transparent rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />

        <div className="relative max-w-4xl mx-auto px-6 py-16">
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-1.5 px-3 py-1 bg-fresh-teal/10 rounded-full">
              <Shield size={14} className="text-fresh-teal" />
              <span className="text-small font-medium text-fresh-teal">Privacy Policy</span>
            </div>
          </div>
          <h1 className="text-display font-bold text-text-primary mb-4 tracking-tight">
            Privacy Policy
          </h1>
          <p className="text-h3 text-text-secondary font-normal max-w-2xl leading-relaxed">
            Your privacy matters to us. This policy explains how Data Flow Canvas handles your data.
          </p>
          <p className="text-small text-text-muted mt-4">
            Last updated: {LAST_UPDATED}
          </p>
        </div>
      </section>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Privacy Highlight */}
        <div className="bg-fresh-teal/10 border border-fresh-teal/20 rounded-2xl p-6 mb-12">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-fresh-teal/20 rounded-xl">
              <Lock size={24} className="text-fresh-teal" />
            </div>
            <div>
              <h3 className="text-h3 font-semibold text-text-primary mb-2">Privacy-First Design</h3>
              <p className="text-text-secondary">
                Data Flow Canvas is designed with privacy at its core. All data processing happens
                locally in your browser using WebAssembly. Your project data never leaves your device
                and is stored only in your browser's local storage.
              </p>
            </div>
          </div>
        </div>

        <Section icon={Database} title="Information We Collect" color="#6366f1">
          <p>
            When you create an account, we collect the following information through our
            authentication provider (Supabase):
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li><strong>Email address</strong> - Used for account identification and communication</li>
            <li><strong>Password</strong> - Securely hashed and stored by Supabase</li>
            <li><strong>First name and last name</strong> - Used to personalize your experience</li>
            <li><strong>Company name</strong> (optional) - If you choose to provide it</li>
          </ul>
        </Section>

        <Section icon={Eye} title="How We Use Your Information" color="#8b5cf6">
          <p>We use your information solely for:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Account authentication and management</li>
            <li>Enabling real-time collaboration features (when you choose to use them)</li>
            <li>Communicating important service updates</li>
          </ul>
          <p>
            We do <strong>not</strong> use your information for advertising, profiling, or any
            purpose other than operating the service.
          </p>
        </Section>

        <Section icon={Server} title="Data Storage & Processing" color="#14b8a6">
          <p><strong>Your Project Data (Pipelines, Datasets)</strong></p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Stored locally in your browser's IndexedDB - never sent to our servers</li>
            <li>All data processing happens in your browser using Python (Pyodide/WebAssembly)</li>
            <li>Your data remains on your device at all times</li>
          </ul>

          <p className="mt-4"><strong>Authentication Data</strong></p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Stored securely by Supabase (our authentication provider)</li>
            <li>Supabase maintains SOC 2 Type II compliance</li>
            <li>For more information, see{' '}
              <a
                href="https://supabase.com/privacy"
                target="_blank"
                rel="noopener noreferrer"
                className="text-electric-indigo hover:underline"
              >
                Supabase's Privacy Policy
              </a>
            </li>
          </ul>
        </Section>

        <Section icon={Globe} title="Third-Party Services" color="#f59e0b">
          <p>Data Flow Canvas uses the following third-party services:</p>
          <div className="overflow-x-auto">
            <table className="w-full mt-4 text-sm">
              <thead>
                <tr className="border-b border-border-default">
                  <th className="text-left py-2 text-text-primary font-semibold">Service</th>
                  <th className="text-left py-2 text-text-primary font-semibold">Purpose</th>
                  <th className="text-left py-2 text-text-primary font-semibold">Data Shared</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-default">
                <tr>
                  <td className="py-3">Supabase</td>
                  <td className="py-3">Authentication</td>
                  <td className="py-3">Email, password, profile info</td>
                </tr>
                <tr>
                  <td className="py-3">Y.js Signaling Server</td>
                  <td className="py-3">Peer discovery for collaboration</td>
                  <td className="py-3">Session ID only (no personal data)</td>
                </tr>
                <tr>
                  <td className="py-3">Google Fonts</td>
                  <td className="py-3">Typography</td>
                  <td className="py-3">None</td>
                </tr>
                <tr>
                  <td className="py-3">jsDelivr CDN</td>
                  <td className="py-3">Pyodide runtime delivery</td>
                  <td className="py-3">None</td>
                </tr>
                <tr>
                  <td className="py-3">STUN Servers</td>
                  <td className="py-3">WebRTC connection setup</td>
                  <td className="py-3">Connection metadata only</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Section>

        <Section icon={Shield} title="Cookies & Tracking" color="#14b8a6">
          <p>
            <strong>We do not use cookies for tracking.</strong> We do not use any analytics
            services such as Google Analytics, Mixpanel, or similar tools.
          </p>
          <p>
            The only browser storage we use is IndexedDB for storing your project data locally
            on your device.
          </p>
        </Section>

        <Section icon={Users} title="Real-Time Collaboration" color="#8b5cf6">
          <p>
            When you use collaboration features, connections are established directly between
            participants using WebRTC (peer-to-peer). This means:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Collaboration data flows directly between users, not through our servers</li>
            <li>Only your display name and cursor position are shared with collaborators</li>
            <li>The Y.js signaling server only facilitates initial peer discovery</li>
          </ul>
        </Section>

        <Section icon={Trash2} title="Your Rights" color="#f43f5e">
          <p>You have the right to:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li><strong>Access</strong> - Request a copy of your personal data</li>
            <li><strong>Correction</strong> - Update your account information at any time</li>
            <li><strong>Deletion</strong> - Request deletion of your account and associated data</li>
            <li><strong>Export</strong> - Export your project data using the Export CSV block</li>
          </ul>
          <p className="mt-4">
            To exercise these rights, please open an issue on our{' '}
            <a
              href={`${GITHUB_REPO}/issues`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-electric-indigo hover:underline"
            >
              GitHub repository
            </a>
            .
          </p>
        </Section>

        <Section icon={FileText} title="Data Retention" color="#6366f1">
          <p>
            <strong>Account data:</strong> We retain your account information as long as your
            account is active. Upon account deletion, your data will be removed within 30 days.
          </p>
          <p>
            <strong>Project data:</strong> Since project data is stored locally in your browser,
            you have full control over its retention. Clear your browser data to remove it.
          </p>
        </Section>

        <Section icon={Globe} title="International Users" color="#f59e0b">
          <p>
            Data Flow Canvas is operated from Japan. By using our service, you acknowledge that
            your authentication data may be processed in regions where Supabase operates its
            infrastructure.
          </p>
          <p>
            For users in the European Economic Area (EEA), Supabase provides appropriate
            safeguards for international data transfers.
          </p>
        </Section>

        <Section icon={Mail} title="Contact Us" color="#14b8a6">
          <p>
            For privacy-related questions or concerns, please open an issue on our GitHub repository:
          </p>
          <a
            href={`${GITHUB_REPO}/issues`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 mt-2 text-electric-indigo hover:underline"
          >
            <Mail size={16} />
            {GITHUB_REPO}/issues
          </a>
        </Section>

        <Section icon={FileText} title="Changes to This Policy" color="#8b5cf6">
          <p>
            We may update this privacy policy from time to time. We will notify users of any
            material changes by posting the new policy on this page with an updated revision date.
          </p>
        </Section>

        {/* Footer */}
        <footer className="text-center py-12 border-t border-border-default mt-12">
          <div className="flex items-center justify-center gap-2 mb-4">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-10 h-10 rounded-xl shadow-lg" />
          </div>
          <p className="text-text-secondary font-medium mb-1">
            Data Flow Canvas
          </p>
          <p className="text-text-muted text-small">
            &copy; 2025 Lavelle Hatcher Jr. All rights reserved.
          </p>
          <div className="flex items-center justify-center gap-4 mt-4 text-small">
            <Link to="/terms" className="text-text-muted hover:text-text-primary transition-colors">
              Terms of Service
            </Link>
            <span className="text-text-muted">|</span>
            <Link to="/privacy" className="text-electric-indigo font-medium">
              Privacy Policy
            </Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
