/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { Plus, FileText, Trash2, Clock, Layers } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { useProjectStore } from '@/stores/projectStore';
import { cn } from '@/lib/utils/cn';

export default function ProjectsPage() {
  const { t } = useTranslation();
  const { projectList } = useProjectStore();

  const sampleProjects = [
    {
      id: 'iris-analysis',
      nameKey: 'projects.samples.irisClassification.name',
      descriptionKey: 'projects.samples.irisClassification.description',
      icon: Layers,
      bgColor: 'bg-electric-indigo/20',
      iconColor: 'text-electric-indigo',
    },
    {
      id: 'data-cleaning',
      nameKey: 'projects.samples.dataCleaningPipeline.name',
      descriptionKey: 'projects.samples.dataCleaningPipeline.description',
      icon: FileText,
      bgColor: 'bg-fresh-teal/20',
      iconColor: 'text-fresh-teal',
    },
    {
      id: 'regression',
      nameKey: 'projects.samples.linearRegression.name',
      descriptionKey: 'projects.samples.linearRegression.description',
      icon: Layers,
      bgColor: 'bg-soft-violet/20',
      iconColor: 'text-soft-violet',
    },
  ];

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="border-b border-border-default bg-bg-secondary">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="flex items-center gap-3">
              <img src="/logo.svg" alt="Data Flow Canvas" className="w-10 h-10 rounded-xl" />
              <span className="text-xl font-semibold text-text-primary">
                Data Flow Canvas
              </span>
            </Link>
          </div>

          <div className="flex items-center gap-4">
            <LanguageSelector variant="full" />
            <Link to="/editor">
              <Button variant="primary" leftIcon={<Plus size={18} />}>
                {t('projects.newProject')}
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-h1 text-text-primary mb-2">{t('projects.yourProjects')}</h1>
          <p className="text-text-secondary">
            {t('projects.storedLocally')}
          </p>
        </div>

        {projectList.length === 0 ? (
          <EmptyState t={t} />
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projectList.map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <ProjectCard project={project} t={t} />
              </motion.div>
            ))}
          </div>
        )}

        {/* Sample Projects */}
        <div className="mt-12">
          <h2 className="text-h2 text-text-primary mb-4">{t('projects.examplePipelines')}</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sampleProjects.map((sample, index) => (
              <motion.div
                key={sample.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <SampleProjectCard sample={sample} t={t} />
              </motion.div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

function EmptyState({ t }: { t: (key: string) => string }) {
  return (
    <Card className="max-w-md mx-auto text-center py-12">
      <div className="w-16 h-16 rounded-2xl bg-bg-tertiary flex items-center justify-center mx-auto mb-4">
        <FileText size={32} className="text-text-muted" />
      </div>
      <h3 className="text-h3 text-text-primary mb-2">{t('projects.noProjectsYet')}</h3>
      <p className="text-text-secondary mb-6">
        {t('projects.createFirstProject')}
      </p>
      <Link to="/editor">
        <Button variant="primary" leftIcon={<Plus size={18} />}>
          {t('projects.createProject')}
        </Button>
      </Link>
    </Card>
  );
}

function ProjectCard({ project, t }: { project: any; t: (key: string) => string }) {
  const { removeFromProjectList } = useProjectStore();

  return (
    <Card
      className={cn(
        'group hover:border-electric-indigo/50 transition-all cursor-pointer',
        'border border-border-default'
      )}
    >
      <Link to={`/editor/${project.id}`}>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="truncate">{project.name}</span>
            <button
              onClick={(e) => {
                e.preventDefault();
                removeFromProjectList(project.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-warm-coral/20 text-text-muted hover:text-warm-coral transition-all"
            >
              <Trash2 size={16} />
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 text-small text-text-muted">
            <div className="flex items-center gap-1">
              <Layers size={14} />
              <span>{project.blockCount} {t('projects.blocks')}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock size={14} />
              <span>
                {new Date(project.updatedAt).toLocaleDateString()}
              </span>
            </div>
          </div>
        </CardContent>
      </Link>
    </Card>
  );
}

function SampleProjectCard({ sample, t }: { sample: any; t: (key: string) => string }) {
  return (
    <Card
      className={cn(
        'group hover:border-electric-indigo/50 transition-all cursor-pointer',
        'border border-border-default bg-gradient-to-b from-bg-secondary to-bg-tertiary/50'
      )}
    >
      <Link to={`/editor?sample=${sample.id}`}>
        <CardHeader>
          <div
            className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center mb-2',
              sample.bgColor
            )}
          >
            <sample.icon size={20} className={sample.iconColor} />
          </div>
          <CardTitle>{t(sample.nameKey)}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-small text-text-muted">{t(sample.descriptionKey)}</p>
        </CardContent>
      </Link>
    </Card>
  );
}
