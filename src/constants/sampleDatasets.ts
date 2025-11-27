/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

export interface SampleDataset {
  id: string;
  name: string;
  description: string;
  rows: number;
  columns: number;
}

export const sampleDatasets: SampleDataset[] = [
  {
    id: 'iris',
    name: 'Iris Dataset',
    description: 'Classic dataset for classification with 3 species of iris flowers',
    rows: 150,
    columns: 5,
  },
  {
    id: 'wine',
    name: 'Wine Dataset',
    description: 'Chemical analysis of wines from three different cultivars',
    rows: 178,
    columns: 14,
  },
  {
    id: 'space_missions',
    name: 'Space Missions',
    description: 'Fictional space exploration missions data',
    rows: 400,
    columns: 7,
  },
];
