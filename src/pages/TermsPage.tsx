/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import {
  ChevronLeft,
  FileText,
  Scale,
  User,
  Shield,
  Code2,
  AlertTriangle,
  XCircle,
  RefreshCw,
  Globe,
  Mail,
  Gavel,
  CheckCircle,
  Database,
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

export default function TermsPage() {
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
        <div className="absolute inset-0 bg-gradient-to-br from-electric-indigo/5 via-transparent to-soft-violet/5" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-electric-indigo/10 to-transparent rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />

        <div className="relative max-w-4xl mx-auto px-6 py-16">
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-1.5 px-3 py-1 bg-electric-indigo/10 rounded-full">
              <Scale size={14} className="text-electric-indigo" />
              <span className="text-small font-medium text-electric-indigo">Legal</span>
            </div>
          </div>
          <h1 className="text-display font-bold text-text-primary mb-4 tracking-tight">
            Terms of Service
          </h1>
          <p className="text-h3 text-text-secondary font-normal max-w-2xl leading-relaxed">
            Please read these terms carefully before using Data Flow Canvas.
          </p>
          <p className="text-small text-text-muted mt-4">
            Last updated: {LAST_UPDATED}
          </p>
        </div>
      </section>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        <Section icon={FileText} title="Agreement to Terms" color="#6366f1">
          <p>
            By accessing or using Data Flow Canvas ("the Service"), you agree to be bound by these
            Terms of Service. If you do not agree to these terms, please do not use the Service.
          </p>
          <p>
            These terms apply to all users of the Service, including visitors, registered users,
            and collaborators.
          </p>
        </Section>

        <Section icon={CheckCircle} title="Description of Service" color="#14b8a6">
          <p>
            Data Flow Canvas is a browser-based visual data pipeline builder that allows you to:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Create and manage data transformation pipelines visually</li>
            <li>Process data locally in your browser using Python (via WebAssembly)</li>
            <li>Collaborate with others in real-time through secure server-relayed connections</li>
            <li>Export and visualize your data</li>
          </ul>
          <p>
            All data processing occurs locally in your browser. Your project data is stored in
            your browser's local storage and is not transmitted to our servers.
          </p>
        </Section>

        <Section icon={User} title="User Accounts" color="#8b5cf6">
          <p>To use certain features of the Service, you must create an account. You agree to:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Provide accurate and complete registration information</li>
            <li>Maintain the security of your password and account</li>
            <li>Accept responsibility for all activities under your account</li>
            <li>Notify us immediately of any unauthorized use of your account</li>
          </ul>
          <p>
            You must be at least 13 years old to create an account. If you are under 18, you
            represent that you have your parent or guardian's permission to use the Service.
          </p>
        </Section>

        <Section icon={Database} title="Your Data" color="#14b8a6">
          <p><strong>Ownership:</strong> You retain all rights to your data. We do not claim any
            ownership over the data you process or create using the Service.</p>
          <p><strong>Local Storage:</strong> Your project data (pipelines, datasets, configurations)
            is stored locally in your browser's IndexedDB. We do not have access to this data.</p>
          <p><strong>Responsibility:</strong> You are responsible for maintaining backups of your
            data. Since data is stored locally, clearing your browser data will delete your projects.</p>
          <p><strong>Export:</strong> You can export your data at any time using the Export CSV block.</p>
        </Section>

        <Section icon={Shield} title="Acceptable Use" color="#f59e0b">
          <p>You agree not to use the Service to:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Process or store data that violates any applicable law</li>
            <li>Infringe on intellectual property rights of others</li>
            <li>Transmit malware, viruses, or harmful code</li>
            <li>Attempt to gain unauthorized access to any systems</li>
            <li>Harass, abuse, or harm other users</li>
            <li>Use the Service for any illegal purpose</li>
            <li>Interfere with or disrupt the Service</li>
          </ul>
        </Section>

        <Section icon={Code2} title="Open Source License" color="#6366f1">
          <p>
            Data Flow Canvas is open source software licensed under the{' '}
            <strong>GNU Affero General Public License v3.0 (AGPL-3.0)</strong>.
          </p>
          <p>This means:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>You can view, modify, and distribute the source code</li>
            <li>If you modify and deploy the software, you must make your modifications available
              under the same license</li>
            <li>The software is provided "as is" without warranty</li>
          </ul>
          <p>
            The full license text is available at:{' '}
            <a
              href={`${GITHUB_REPO}/blob/main/LICENSE`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-electric-indigo hover:underline"
            >
              {GITHUB_REPO}/blob/main/LICENSE
            </a>
          </p>
          <p className="mt-4">
            For commercial licensing inquiries (if you need to use the software without AGPL
            obligations), please contact the author through GitHub.
          </p>
        </Section>

        <Section icon={AlertTriangle} title="Disclaimer of Warranties" color="#f43f5e">
          <p>
            THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND,
            EITHER EXPRESS OR IMPLIED.
          </p>
          <p>We do not warrant that:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>The Service will be uninterrupted or error-free</li>
            <li>The results obtained from the Service will be accurate or reliable</li>
            <li>Any errors in the Service will be corrected</li>
          </ul>
          <p>
            You use the Service at your own risk. You are solely responsible for any damage to
            your computer system or loss of data that results from using the Service.
          </p>
        </Section>

        <Section icon={Scale} title="Limitation of Liability" color="#8b5cf6">
          <p>
            TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL DATA FLOW CANVAS, ITS
            AUTHOR, OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL,
            OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Loss of profits, data, or use</li>
            <li>Business interruption</li>
            <li>Cost of substitute services</li>
          </ul>
          <p>
            This limitation applies regardless of the theory of liability (contract, tort, or otherwise),
            even if we have been advised of the possibility of such damages.
          </p>
        </Section>

        <Section icon={XCircle} title="Termination" color="#f43f5e">
          <p>
            We reserve the right to suspend or terminate your access to the Service at any time,
            with or without cause, with or without notice.
          </p>
          <p>You may terminate your account at any time by:</p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Requesting account deletion through our GitHub issues</li>
            <li>Ceasing to use the Service</li>
          </ul>
          <p>
            Upon termination, your right to use the Service will immediately cease. Your locally
            stored data will remain in your browser unless you clear it.
          </p>
        </Section>

        <Section icon={RefreshCw} title="Changes to Terms" color="#f59e0b">
          <p>
            We reserve the right to modify these terms at any time. We will notify users of any
            material changes by:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Posting the updated terms on this page</li>
            <li>Updating the "Last updated" date</li>
          </ul>
          <p>
            Your continued use of the Service after any changes constitutes acceptance of the
            new terms.
          </p>
        </Section>

        <Section icon={Gavel} title="Governing Law" color="#6366f1">
          <p>
            These Terms shall be governed by and construed in accordance with the laws of Japan,
            without regard to its conflict of law provisions.
          </p>
          <p>
            Any disputes arising from these terms or your use of the Service shall be subject to
            the exclusive jurisdiction of the courts of Japan.
          </p>
        </Section>

        <Section icon={Globe} title="International Use" color="#14b8a6">
          <p>
            The Service is operated from Japan. If you access the Service from outside Japan, you
            do so at your own risk and are responsible for compliance with local laws.
          </p>
          <p>
            We make no representation that the Service is appropriate or available for use in
            any particular location.
          </p>
        </Section>

        <Section icon={Mail} title="Contact" color="#8b5cf6">
          <p>
            For questions about these Terms of Service, please open an issue on our GitHub repository:
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
            <Link to="/terms" className="text-electric-indigo font-medium">
              Terms of Service
            </Link>
            <span className="text-text-muted">|</span>
            <Link to="/privacy" className="text-text-muted hover:text-text-primary transition-colors">
              Privacy Policy
            </Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
