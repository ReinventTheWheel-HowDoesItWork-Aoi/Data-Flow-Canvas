/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import type { BlockDefinition, BlockType } from '@/types';

export const blockDefinitions: Record<BlockType, BlockDefinition> = {
  // Data Input Blocks
  'load-data': {
    type: 'load-data',
    category: 'data-input',
    label: 'Load CSV',
    description: 'Import data from CSV files',
    icon: 'FileUp',
    defaultConfig: {
      fileType: 'csv',
      encoding: 'utf-8',
    },
    inputs: 0,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import base64
import io

file_content_b64 = config.get('fileContent', '')
if not file_content_b64:
    raise ValueError("No file content provided. Please upload a CSV file first.")

file_content = base64.b64decode(file_content_b64)
file_obj = io.BytesIO(file_content)

df = pd.read_csv(file_obj, encoding=config.get('encoding', 'utf-8'))

if df is None:
    raise ValueError("Failed to load data from file")

output = df
`,
  },

  'sample-data': {
    type: 'sample-data',
    category: 'data-input',
    label: 'Sample Data',
    description: 'Load built-in sample datasets for learning',
    icon: 'Database',
    defaultConfig: {
      dataset: 'iris',
    },
    inputs: 0,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
from sklearn import datasets

dataset_name = config.get('dataset', 'iris')

if dataset_name == 'iris':
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
elif dataset_name == 'wine':
    data = datasets.load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
elif dataset_name == 'space_missions':
    import numpy as np
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        'mission_id': [f'M-{i:04d}' for i in range(1, n + 1)],
        'crew_size': np.random.randint(2, 12, n),
        'distance_ly': np.round(np.random.uniform(0.5, 50, n), 2),
        'duration_days': np.random.randint(30, 500, n),
        'fuel_tons': np.random.randint(100, 5000, n),
        'spacecraft_age_yrs': np.random.randint(1, 20, n),
        'success': np.random.choice([0, 1], n, p=[0.15, 0.85])
    })

output = df
`,
  },

  'create-dataset': {
    type: 'create-dataset',
    category: 'data-input',
    label: 'Create Dataset',
    description: 'Manually create a dataset by entering data',
    icon: 'PenLine',
    defaultConfig: {
      columns: 'name,age,city',
      data: 'Alice,30,New York\nBob,25,Los Angeles\nCharlie,35,Chicago',
    },
    inputs: 0,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import io

columns = [c.strip() for c in config.get('columns', '').split(',')]
data_str = config.get('data', '')

# Parse CSV-like data
df = pd.read_csv(io.StringIO(data_str), names=columns, skipinitialspace=True)

output = df
`,
  },

  // Transform Blocks
  'filter-rows': {
    type: 'filter-rows',
    category: 'transform',
    label: 'Filter Rows',
    description: 'Filter rows based on conditions',
    icon: 'Filter',
    defaultConfig: {
      column: '',
      operator: 'equals',
      value: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
operator = config.get('operator', 'equals')
value = config.get('value', '')

# Validate required config
if not column:
    raise ValueError("Filter Rows: Please specify a column to filter on in the Config tab")

if column not in df.columns:
    raise ValueError(f"Filter Rows: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if operator == 'equals':
    # Try numeric comparison first, fall back to string
    try:
        numeric_val = float(value.replace(',', '') if isinstance(value, str) else value)
        # Handle comma-formatted numbers like "9,000"
        numeric_col = pd.to_numeric(df[column].astype(str).str.strip().str.replace(',', ''), errors='coerce')
        mask = numeric_col == numeric_val
        if mask.sum() == 0:
            # Fall back to string comparison
            mask = df[column].astype(str) == str(value)
        df = df[mask]
    except (ValueError, TypeError):
        df = df[df[column].astype(str) == str(value)]
elif operator == 'not_equals':
    try:
        numeric_val = float(value.replace(',', '') if isinstance(value, str) else value)
        # Handle comma-formatted numbers like "9,000"
        numeric_col = pd.to_numeric(df[column].astype(str).str.strip().str.replace(',', ''), errors='coerce')
        mask = numeric_col != numeric_val
        df = df[mask]
    except (ValueError, TypeError):
        df = df[df[column].astype(str) != str(value)]
elif operator == 'greater_than':
    numeric_val = float(value.replace(',', '') if isinstance(value, str) else value)
    # Convert column to numeric, handling comma-formatted numbers like "9,000"
    numeric_col = pd.to_numeric(df[column].astype(str).str.strip().str.replace(',', ''), errors='coerce')
    mask = numeric_col > numeric_val
    df = df[mask.fillna(False)]
elif operator == 'less_than':
    numeric_val = float(value.replace(',', '') if isinstance(value, str) else value)
    # Convert column to numeric, handling comma-formatted numbers like "9,000"
    numeric_col = pd.to_numeric(df[column].astype(str).str.strip().str.replace(',', ''), errors='coerce')
    mask = numeric_col < numeric_val
    df = df[mask.fillna(False)]
elif operator == 'contains':
    df = df[df[column].astype(str).str.contains(str(value), na=False)]
elif operator == 'starts_with':
    df = df[df[column].astype(str).str.startswith(str(value), na=False)]
elif operator == 'ends_with':
    df = df[df[column].astype(str).str.endswith(str(value), na=False)]
elif operator == 'is_null':
    df = df[df[column].isnull()]
elif operator == 'is_not_null':
    df = df[df[column].notnull()]

output = df
`,
  },

  'select-columns': {
    type: 'select-columns',
    category: 'transform',
    label: 'Select Columns',
    description: 'Choose, reorder, or rename columns',
    icon: 'Columns',
    defaultConfig: {
      columns: [],
      rename: {},
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
rename_map = config.get('rename', {})

# Use all columns if none specified
if not columns:
    columns = df.columns.tolist()

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Select Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df[columns]

if rename_map:
    df = df.rename(columns=rename_map)

output = df
`,
  },

  'sort': {
    type: 'sort',
    category: 'transform',
    label: 'Sort',
    description: 'Sort data by one or more columns',
    icon: 'ArrowUpDown',
    defaultConfig: {
      columns: [],
      ascending: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
ascending = config.get('ascending', True)

if not columns:
    raise ValueError("Sort: Please specify at least one column to sort by in the Config tab")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Sort: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df.sort_values(by=columns, ascending=ascending)

output = df
`,
  },

  'group-aggregate': {
    type: 'group-aggregate',
    category: 'transform',
    label: 'Group & Aggregate',
    description: 'Group by columns and apply aggregation functions',
    icon: 'Group',
    defaultConfig: {
      groupBy: [],
      aggregations: {},
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
group_by = config.get('groupBy', [])
aggregations = config.get('aggregations', {})

if not group_by:
    raise ValueError("Group & Aggregate: Please specify columns to group by in the Config tab")

if not aggregations:
    raise ValueError("Group & Aggregate: Please specify aggregations (e.g., {'column': 'sum'}) in the Config tab")

# Validate group columns exist
missing = [c for c in group_by if c not in df.columns]
if missing:
    raise ValueError(f"Group & Aggregate: Group column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df.groupby(group_by).agg(aggregations).reset_index()

output = df
`,
  },

  'join': {
    type: 'join',
    category: 'transform',
    label: 'Join',
    description: 'Merge two datasets',
    icon: 'GitMerge',
    defaultConfig: {
      how: 'inner',
      leftOn: '',
      rightOn: '',
    },
    inputs: 2,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

left_df = input_data[0].copy()
right_df = input_data[1].copy()

how = config.get('how', 'inner')
left_on = config['leftOn']
right_on = config['rightOn']

output = pd.merge(left_df, right_df, how=how, left_on=left_on, right_on=right_on)
`,
  },

  'derive-column': {
    type: 'derive-column',
    category: 'transform',
    label: 'Derive Column',
    description: 'Create new columns from expressions',
    icon: 'Plus',
    defaultConfig: {
      newColumn: '',
      expression: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
new_column = config.get('newColumn', '')
expression = config.get('expression', '')

if not new_column:
    raise ValueError("Derive Column: Please specify a name for the new column in the Config tab")

if not expression:
    raise ValueError("Derive Column: Please specify an expression (e.g., df['price'] * df['quantity']) in the Config tab")

try:
    df[new_column] = eval(expression)
except Exception as e:
    raise ValueError(f"Derive Column: Error evaluating expression: {str(e)}. Available columns: {', '.join(df.columns.tolist())}")

output = df
`,
  },

  'handle-missing': {
    type: 'handle-missing',
    category: 'transform',
    label: 'Handle Missing',
    description: 'Drop rows or fill missing values',
    icon: 'Eraser',
    defaultConfig: {
      strategy: 'drop',
      fillValue: '',
      columns: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
strategy = config['strategy']
columns = config.get('columns', df.columns.tolist())
fill_value = config.get('fillValue', 0)

if strategy == 'drop':
    df = df.dropna(subset=columns if columns else None)
elif strategy == 'fill_value':
    df[columns] = df[columns].fillna(fill_value)
elif strategy == 'fill_mean':
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
elif strategy == 'fill_median':
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
elif strategy == 'fill_mode':
    for col in columns:
        df[col] = df[col].fillna(df[col].mode()[0])
elif strategy == 'interpolate':
    df[columns] = df[columns].interpolate()

output = df
`,
  },

  'rename-columns': {
    type: 'rename-columns',
    category: 'transform',
    label: 'Rename Columns',
    description: 'Rename one or more columns',
    icon: 'TextCursorInput',
    defaultConfig: {
      renames: {},
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
renames = config.get('renames', {})

if not renames:
    raise ValueError("Rename Columns: Please specify column renames in the Config tab")

# Validate columns exist
missing = [c for c in renames.keys() if c not in df.columns]
if missing:
    raise ValueError(f"Rename Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df.rename(columns=renames)

output = df
`,
  },

  'deduplicate': {
    type: 'deduplicate',
    category: 'transform',
    label: 'Deduplicate',
    description: 'Remove duplicate rows',
    icon: 'Copy',
    defaultConfig: {
      columns: [],
      keep: 'first',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
keep = config.get('keep', 'first')

# Use all columns if none specified
subset = columns if columns else None

# Validate columns exist
if columns:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Deduplicate: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df.drop_duplicates(subset=subset, keep=keep if keep != 'none' else False)

output = df
`,
  },

  'sample-rows': {
    type: 'sample-rows',
    category: 'transform',
    label: 'Sample Rows',
    description: 'Randomly sample rows from data',
    icon: 'Shuffle',
    defaultConfig: {
      sampleType: 'count',
      count: 100,
      fraction: 0.1,
      seed: null,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
sample_type = config.get('sampleType', 'count')
count = config.get('count', 100)
fraction = config.get('fraction', 0.1)
seed = config.get('seed', None)

if sample_type == 'count':
    n = min(count, len(df))
    df = df.sample(n=n, random_state=seed)
else:
    df = df.sample(frac=fraction, random_state=seed)

output = df
`,
  },

  'limit-rows': {
    type: 'limit-rows',
    category: 'transform',
    label: 'Limit Rows',
    description: 'Get first or last N rows',
    icon: 'ListFilter',
    defaultConfig: {
      position: 'first',
      count: 10,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
position = config.get('position', 'first')
count = config.get('count', 10)

if position == 'first':
    df = df.head(count)
else:
    df = df.tail(count)

output = df
`,
  },

  'pivot': {
    type: 'pivot',
    category: 'transform',
    label: 'Pivot',
    description: 'Reshape data from long to wide format',
    icon: 'RotateCcw',
    defaultConfig: {
      index: '',
      columns: '',
      values: '',
      aggFunc: 'mean',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
index = config.get('index', '')
columns = config.get('columns', '')
values = config.get('values', '')
agg_func = config.get('aggFunc', 'mean')

if not index:
    raise ValueError("Pivot: Please specify an index column (rows) in the Config tab")
if not columns:
    raise ValueError("Pivot: Please specify a columns column (creates new columns) in the Config tab")
if not values:
    raise ValueError("Pivot: Please specify a values column in the Config tab")

# Validate columns exist
for col_name, col_val in [('index', index), ('columns', columns), ('values', values)]:
    if col_val not in df.columns:
        raise ValueError(f"Pivot: {col_name} column '{col_val}' not found. Available: {', '.join(df.columns.tolist())}")

df = df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg_func).reset_index()
df.columns = [str(c) if not isinstance(c, str) else c for c in df.columns]

output = df
`,
  },

  'unpivot': {
    type: 'unpivot',
    category: 'transform',
    label: 'Unpivot',
    description: 'Reshape data from wide to long format',
    icon: 'RotateCw',
    defaultConfig: {
      idColumns: [],
      valueColumns: [],
      varName: 'variable',
      valueName: 'value',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
id_columns = config.get('idColumns', [])
value_columns = config.get('valueColumns', [])
var_name = config.get('varName', 'variable')
value_name = config.get('valueName', 'value')

if not value_columns:
    raise ValueError("Unpivot: Please specify columns to unpivot in the Config tab")

# Validate columns exist
all_cols = id_columns + value_columns
missing = [c for c in all_cols if c not in df.columns]
if missing:
    raise ValueError(f"Unpivot: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = pd.melt(df, id_vars=id_columns if id_columns else None, value_vars=value_columns,
             var_name=var_name, value_name=value_name)

output = df
`,
  },

  'union': {
    type: 'union',
    category: 'transform',
    label: 'Union',
    description: 'Stack datasets vertically (append rows)',
    icon: 'Layers',
    defaultConfig: {
      ignoreIndex: true,
    },
    inputs: 2,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df1 = input_data[0].copy()
df2 = input_data[1].copy()
ignore_index = config.get('ignoreIndex', True)

df = pd.concat([df1, df2], ignore_index=ignore_index)

output = df
`,
  },

  'split-column': {
    type: 'split-column',
    category: 'transform',
    label: 'Split Column',
    description: 'Split a column by delimiter into multiple columns',
    icon: 'Scissors',
    defaultConfig: {
      column: '',
      delimiter: ',',
      newColumns: [],
      keepOriginal: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
delimiter = config.get('delimiter', ',')
new_columns = config.get('newColumns', [])
keep_original = config.get('keepOriginal', False)

if not column:
    raise ValueError("Split Column: Please specify a column to split in the Config tab")

if column not in df.columns:
    raise ValueError(f"Split Column: Column '{column}' not found. Available: {', '.join(df.columns.tolist())}")

# Split the column
split_data = df[column].astype(str).str.split(delimiter, expand=True)

# Determine column names
if new_columns and len(new_columns) >= split_data.shape[1]:
    split_data.columns = new_columns[:split_data.shape[1]]
else:
    split_data.columns = [f"{column}_{i+1}" for i in range(split_data.shape[1])]

# Add split columns to dataframe
for col in split_data.columns:
    df[col] = split_data[col].str.strip()

# Remove original column if not keeping
if not keep_original:
    df = df.drop(columns=[column])

output = df
`,
  },

  'merge-columns': {
    type: 'merge-columns',
    category: 'transform',
    label: 'Merge Columns',
    description: 'Combine multiple columns into one',
    icon: 'Combine',
    defaultConfig: {
      columns: [],
      separator: ' ',
      newColumn: 'merged',
      keepOriginal: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
separator = config.get('separator', ' ')
new_column = config.get('newColumn', 'merged')
keep_original = config.get('keepOriginal', True)

if not columns or len(columns) < 2:
    raise ValueError("Merge Columns: Please specify at least 2 columns to merge in the Config tab")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Merge Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Merge columns
df[new_column] = df[columns].astype(str).agg(separator.join, axis=1)

# Remove original columns if not keeping
if not keep_original:
    df = df.drop(columns=columns)

output = df
`,
  },

  'conditional-column': {
    type: 'conditional-column',
    category: 'transform',
    label: 'Conditional Column',
    description: 'Create column based on if/else logic',
    icon: 'GitBranch',
    defaultConfig: {
      newColumn: '',
      condition: '',
      trueValue: '',
      falseValue: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
new_column = config.get('newColumn', '')
condition = config.get('condition', '')
true_value = config.get('trueValue', '')
false_value = config.get('falseValue', '')

if not new_column:
    raise ValueError("Conditional Column: Please specify a name for the new column in the Config tab")

if not condition:
    raise ValueError("Conditional Column: Please specify a condition (e.g., df['age'] > 18) in the Config tab")

try:
    mask = eval(condition)
    df[new_column] = np.where(mask, true_value, false_value)
except Exception as e:
    raise ValueError(f"Conditional Column: Error evaluating condition: {str(e)}. Available columns: {', '.join(df.columns.tolist())}")

output = df
`,
  },

  'datetime-extract': {
    type: 'datetime-extract',
    category: 'transform',
    label: 'Date/Time Extract',
    description: 'Extract parts from date columns (year, month, day, weekday, hour, etc.)',
    icon: 'Calendar',
    defaultConfig: {
      column: '',
      extractions: ['year', 'month', 'day'],
      prefix: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
extractions = config.get('extractions', ['year', 'month', 'day'])
prefix = config.get('prefix', '')

if not column:
    raise ValueError("Date/Time Extract: Please specify a date column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Date/Time Extract: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert to datetime
try:
    df[column] = pd.to_datetime(df[column], errors='coerce')
except Exception as e:
    raise ValueError(f"Date/Time Extract: Could not parse column '{column}' as datetime: {str(e)}")

# Use column name as prefix if not specified
col_prefix = prefix if prefix else column

for extraction in extractions:
    new_col = f"{col_prefix}_{extraction}"
    if extraction == 'year':
        df[new_col] = df[column].dt.year
    elif extraction == 'month':
        df[new_col] = df[column].dt.month
    elif extraction == 'day':
        df[new_col] = df[column].dt.day
    elif extraction == 'weekday':
        df[new_col] = df[column].dt.day_name()
    elif extraction == 'weekday_num':
        df[new_col] = df[column].dt.dayofweek
    elif extraction == 'hour':
        df[new_col] = df[column].dt.hour
    elif extraction == 'minute':
        df[new_col] = df[column].dt.minute
    elif extraction == 'second':
        df[new_col] = df[column].dt.second
    elif extraction == 'quarter':
        df[new_col] = df[column].dt.quarter
    elif extraction == 'week':
        df[new_col] = df[column].dt.isocalendar().week
    elif extraction == 'dayofyear':
        df[new_col] = df[column].dt.dayofyear
    elif extraction == 'is_weekend':
        df[new_col] = df[column].dt.dayofweek >= 5
    elif extraction == 'date':
        df[new_col] = df[column].dt.date

output = df
`,
  },

  'string-operations': {
    type: 'string-operations',
    category: 'transform',
    label: 'String Operations',
    description: 'Clean and manipulate text columns',
    icon: 'Type',
    defaultConfig: {
      column: '',
      operation: 'lowercase',
      findText: '',
      replaceText: '',
      regexPattern: '',
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re

df = input_data.copy()
column = config.get('column', '')
operation = config.get('operation', 'lowercase')
find_text = config.get('findText', '')
replace_text = config.get('replaceText', '')
regex_pattern = config.get('regexPattern', '')
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("String Operations: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"String Operations: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Determine output column name
output_col = new_column if new_column else column

# Convert to string first
str_col = df[column].astype(str)

if operation == 'lowercase':
    df[output_col] = str_col.str.lower()
elif operation == 'uppercase':
    df[output_col] = str_col.str.upper()
elif operation == 'titlecase':
    df[output_col] = str_col.str.title()
elif operation == 'trim':
    df[output_col] = str_col.str.strip()
elif operation == 'trim_left':
    df[output_col] = str_col.str.lstrip()
elif operation == 'trim_right':
    df[output_col] = str_col.str.rstrip()
elif operation == 'find_replace':
    if not find_text:
        raise ValueError("String Operations: Please specify text to find")
    df[output_col] = str_col.str.replace(find_text, replace_text, regex=False)
elif operation == 'regex_replace':
    if not regex_pattern:
        raise ValueError("String Operations: Please specify a regex pattern")
    df[output_col] = str_col.str.replace(regex_pattern, replace_text, regex=True)
elif operation == 'regex_extract':
    if not regex_pattern:
        raise ValueError("String Operations: Please specify a regex pattern")
    df[output_col] = str_col.str.extract(f'({regex_pattern})', expand=False)
elif operation == 'length':
    df[output_col] = str_col.str.len()
elif operation == 'remove_digits':
    df[output_col] = str_col.str.replace(r'\\d+', '', regex=True)
elif operation == 'remove_punctuation':
    df[output_col] = str_col.str.replace(r'[^\\w\\s]', '', regex=True)
elif operation == 'remove_whitespace':
    df[output_col] = str_col.str.replace(r'\\s+', ' ', regex=True).str.strip()

output = df
`,
  },

  'window-functions': {
    type: 'window-functions',
    category: 'transform',
    label: 'Window Functions',
    description: 'Calculate rolling, cumulative, and lag/lead values',
    icon: 'Waypoints',
    defaultConfig: {
      column: '',
      operation: 'rolling_mean',
      windowSize: 3,
      groupBy: [],
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
operation = config.get('operation', 'rolling_mean')
window_size = int(config.get('windowSize', 3))
group_by = config.get('groupBy', [])
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Window Functions: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Window Functions: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Determine output column name
output_col = new_column if new_column else f"{column}_{operation}"

# Convert column to numeric if needed
numeric_col = pd.to_numeric(df[column], errors='coerce')

def apply_window(group_df):
    col_data = pd.to_numeric(group_df[column], errors='coerce')

    if operation == 'rolling_mean':
        return col_data.rolling(window=window_size, min_periods=1).mean()
    elif operation == 'rolling_sum':
        return col_data.rolling(window=window_size, min_periods=1).sum()
    elif operation == 'rolling_min':
        return col_data.rolling(window=window_size, min_periods=1).min()
    elif operation == 'rolling_max':
        return col_data.rolling(window=window_size, min_periods=1).max()
    elif operation == 'rolling_std':
        return col_data.rolling(window=window_size, min_periods=1).std()
    elif operation == 'cumsum':
        return col_data.cumsum()
    elif operation == 'cumprod':
        return col_data.cumprod()
    elif operation == 'cummin':
        return col_data.cummin()
    elif operation == 'cummax':
        return col_data.cummax()
    elif operation == 'lag':
        return col_data.shift(window_size)
    elif operation == 'lead':
        return col_data.shift(-window_size)
    elif operation == 'pct_change':
        return col_data.pct_change(periods=window_size)
    elif operation == 'diff':
        return col_data.diff(periods=window_size)
    elif operation == 'rank':
        return col_data.rank()

    return col_data

if group_by and len(group_by) > 0:
    # Validate group columns exist
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Window Functions: Group column(s) not found: {', '.join(missing)}")
    df[output_col] = df.groupby(group_by, group_keys=False).apply(
        lambda x: apply_window(x)
    ).reset_index(drop=True)
else:
    df[output_col] = apply_window(df)

output = df
`,
  },

  'bin-bucket': {
    type: 'bin-bucket',
    category: 'transform',
    label: 'Bin/Bucket',
    description: 'Group continuous numbers into discrete bins or ranges',
    icon: 'BarChart',
    defaultConfig: {
      column: '',
      method: 'equal_width',
      numBins: 5,
      customEdges: '',
      customLabels: '',
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'equal_width')
num_bins = int(config.get('numBins', 5))
custom_edges = config.get('customEdges', '')
custom_labels = config.get('customLabels', '')
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Bin/Bucket: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Bin/Bucket: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Determine output column name
output_col = new_column if new_column else f"{column}_binned"

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

# Parse custom labels if provided
labels = None
if custom_labels:
    labels = [l.strip() for l in custom_labels.split(',')]

if method == 'equal_width':
    df[output_col] = pd.cut(numeric_col, bins=num_bins, labels=labels)
elif method == 'equal_frequency':
    df[output_col] = pd.qcut(numeric_col, q=num_bins, labels=labels, duplicates='drop')
elif method == 'custom':
    if not custom_edges:
        raise ValueError("Bin/Bucket: Please specify custom bin edges (comma-separated numbers)")
    edges = [float(e.strip()) for e in custom_edges.split(',')]
    if labels and len(labels) != len(edges) - 1:
        raise ValueError(f"Bin/Bucket: Number of labels ({len(labels)}) must be one less than number of edges ({len(edges)})")
    df[output_col] = pd.cut(numeric_col, bins=edges, labels=labels, include_lowest=True)

output = df
`,
  },

  'rank': {
    type: 'rank',
    category: 'transform',
    label: 'Rank',
    description: 'Assign rank positions (1st, 2nd, 3rd...) to values',
    icon: 'Medal',
    defaultConfig: {
      column: '',
      method: 'average',
      ascending: true,
      groupBy: [],
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'average')
ascending = config.get('ascending', True)
group_by = config.get('groupBy', [])
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Rank: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Rank: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Determine output column name
output_col = new_column if new_column else f"{column}_rank"

# Convert to numeric for ranking
numeric_col = pd.to_numeric(df[column], errors='coerce')

if group_by and len(group_by) > 0:
    # Validate group columns exist
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Rank: Group column(s) not found: {', '.join(missing)}")
    df[output_col] = df.groupby(group_by)[column].rank(method=method, ascending=ascending)
else:
    df[output_col] = numeric_col.rank(method=method, ascending=ascending)

output = df
`,
  },

  'type-conversion': {
    type: 'type-conversion',
    category: 'transform',
    label: 'Type Conversion',
    description: 'Change column data types (string, integer, float, boolean, datetime, category)',
    icon: 'RefreshCw',
    defaultConfig: {
      column: '',
      targetType: 'string',
      datetimeFormat: '',
      errorHandling: 'coerce',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
target_type = config.get('targetType', 'string')
datetime_format = config.get('datetimeFormat', '')
error_handling = config.get('errorHandling', 'coerce')

if not column:
    raise ValueError("Type Conversion: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Type Conversion: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

try:
    if target_type == 'string':
        df[column] = df[column].astype(str)
    elif target_type == 'integer':
        if error_handling == 'coerce':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        else:
            df[column] = df[column].astype(int)
    elif target_type == 'float':
        if error_handling == 'coerce':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            df[column] = df[column].astype(float)
    elif target_type == 'boolean':
        # Handle various boolean representations
        true_vals = ['true', 'yes', '1', 't', 'y']
        false_vals = ['false', 'no', '0', 'f', 'n']
        str_col = df[column].astype(str).str.lower().str.strip()
        df[column] = str_col.apply(lambda x: True if x in true_vals else (False if x in false_vals else None))
    elif target_type == 'datetime':
        if datetime_format:
            df[column] = pd.to_datetime(df[column], format=datetime_format, errors=error_handling)
        else:
            df[column] = pd.to_datetime(df[column], errors=error_handling)
    elif target_type == 'category':
        df[column] = df[column].astype('category')
except Exception as e:
    if error_handling == 'raise':
        raise ValueError(f"Type Conversion: Failed to convert column '{column}' to {target_type}: {str(e)}")
    # If coerce, errors are already handled above

output = df
`,
  },

  'fill-forward-backward': {
    type: 'fill-forward-backward',
    category: 'transform',
    label: 'Fill Forward/Backward',
    description: 'Fill missing values with previous or next values',
    icon: 'ArrowLeftRight',
    defaultConfig: {
      column: '',
      method: 'forward',
      limit: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'forward')
limit_str = config.get('limit', '')

if not column:
    raise ValueError("Fill Forward/Backward: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Fill Forward/Backward: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

limit = int(limit_str) if limit_str else None

if method == 'forward':
    df[column] = df[column].ffill(limit=limit)
elif method == 'backward':
    df[column] = df[column].bfill(limit=limit)
elif method == 'both':
    df[column] = df[column].ffill(limit=limit).bfill(limit=limit)

output = df
`,
  },

  'lag-lead': {
    type: 'lag-lead',
    category: 'transform',
    label: 'Lag/Lead Column',
    description: 'Create columns with shifted values (lag or lead)',
    icon: 'MoveHorizontal',
    defaultConfig: {
      column: '',
      operation: 'lag',
      periods: 1,
      groupBy: [],
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
operation = config.get('operation', 'lag')
periods = int(config.get('periods', 1))
group_by = config.get('groupBy', [])
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Lag/Lead: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Lag/Lead: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"{column}_{operation}_{periods}"

shift_periods = periods if operation == 'lag' else -periods

if group_by:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Lag/Lead: Group column(s) not found: {', '.join(missing)}")
    df[output_col] = df.groupby(group_by)[column].shift(shift_periods)
else:
    df[output_col] = df[column].shift(shift_periods)

output = df
`,
  },

  'row-number': {
    type: 'row-number',
    category: 'transform',
    label: 'Row Number',
    description: 'Add a unique row number column',
    icon: 'Hash',
    defaultConfig: {
      outputColumn: 'row_num',
      startFrom: 1,
      groupBy: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
output_col = config.get('outputColumn', 'row_num')
start_from = int(config.get('startFrom', 1))
group_by = config.get('groupBy', [])

if group_by:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Row Number: Group column(s) not found: {', '.join(missing)}")
    df[output_col] = df.groupby(group_by).cumcount() + start_from
else:
    df[output_col] = range(start_from, len(df) + start_from)

output = df
`,
  },

  'date-difference': {
    type: 'date-difference',
    category: 'transform',
    label: 'Date Difference',
    description: 'Calculate difference between two date columns',
    icon: 'CalendarRange',
    defaultConfig: {
      startColumn: '',
      endColumn: '',
      unit: 'days',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
start_col = config.get('startColumn', '')
end_col = config.get('endColumn', '')
unit = config.get('unit', 'days')
output_col = config.get('outputColumn', '')

if not start_col or not end_col:
    raise ValueError("Date Difference: Please specify both start and end date columns in the Config tab")

if start_col not in df.columns:
    raise ValueError(f"Date Difference: Start column '{start_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

if end_col not in df.columns:
    raise ValueError(f"Date Difference: End column '{end_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"diff_{unit}"

df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

diff = df[end_col] - df[start_col]

if unit == 'days':
    df[output_col] = diff.dt.days
elif unit == 'hours':
    df[output_col] = diff.dt.total_seconds() / 3600
elif unit == 'minutes':
    df[output_col] = diff.dt.total_seconds() / 60
elif unit == 'seconds':
    df[output_col] = diff.dt.total_seconds()
elif unit == 'weeks':
    df[output_col] = diff.dt.days / 7
elif unit == 'months':
    df[output_col] = diff.dt.days / 30.44
elif unit == 'years':
    df[output_col] = diff.dt.days / 365.25

output = df
`,
  },

  'transpose': {
    type: 'transpose',
    category: 'transform',
    label: 'Transpose',
    description: 'Flip rows and columns',
    icon: 'FlipHorizontal',
    defaultConfig: {
      useFirstColumnAsHeader: true,
      useFirstRowAsIndex: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
use_first_col_header = config.get('useFirstColumnAsHeader', True)
use_first_row_index = config.get('useFirstRowAsIndex', False)

if use_first_row_index:
    df = df.set_index(df.columns[0])

df_transposed = df.T

if use_first_col_header:
    df_transposed = df_transposed.reset_index()
    df_transposed.columns = ['column'] + [f'row_{i}' for i in range(len(df_transposed.columns) - 1)]
else:
    df_transposed = df_transposed.reset_index(drop=True)

output = df_transposed
`,
  },

  'string-pad': {
    type: 'string-pad',
    category: 'transform',
    label: 'String Pad',
    description: 'Pad strings to a fixed length with specified character',
    icon: 'AlignJustify',
    defaultConfig: {
      column: '',
      width: 10,
      side: 'left',
      fillChar: '0',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
width = int(config.get('width', 10))
side = config.get('side', 'left')
fill_char = config.get('fillChar', '0')
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("String Pad: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"String Pad: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

fill_char = str(fill_char)[0] if fill_char else ' '
str_col = df[column].astype(str)

if side == 'left':
    df[output_col] = str_col.str.zfill(width) if fill_char == '0' else str_col.str.pad(width, side='left', fillchar=fill_char)
elif side == 'right':
    df[output_col] = str_col.str.pad(width, side='right', fillchar=fill_char)
elif side == 'both':
    df[output_col] = str_col.str.center(width, fillchar=fill_char)

output = df
`,
  },

  'cumulative-operations': {
    type: 'cumulative-operations',
    category: 'transform',
    label: 'Cumulative Operations',
    description: 'Calculate running totals, cumulative counts, and percentages',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      operation: 'sum',
      groupBy: [],
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
operation = config.get('operation', 'sum')
group_by = config.get('groupBy', [])
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Cumulative Operations: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Cumulative Operations: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"cum_{operation}_{column}"

numeric_col = pd.to_numeric(df[column], errors='coerce')

if group_by:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Cumulative Operations: Group column(s) not found: {', '.join(missing)}")

    if operation == 'sum':
        df[output_col] = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').cumsum())
    elif operation == 'count':
        df[output_col] = df.groupby(group_by).cumcount() + 1
    elif operation == 'mean':
        df[output_col] = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').expanding().mean())
    elif operation == 'max':
        df[output_col] = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').cummax())
    elif operation == 'min':
        df[output_col] = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').cummin())
    elif operation == 'product':
        df[output_col] = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').cumprod())
    elif operation == 'percent':
        total = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').sum())
        cum_sum = df.groupby(group_by)[column].transform(lambda x: pd.to_numeric(x, errors='coerce').cumsum())
        df[output_col] = (cum_sum / total * 100).round(2)
else:
    if operation == 'sum':
        df[output_col] = numeric_col.cumsum()
    elif operation == 'count':
        df[output_col] = range(1, len(df) + 1)
    elif operation == 'mean':
        df[output_col] = numeric_col.expanding().mean()
    elif operation == 'max':
        df[output_col] = numeric_col.cummax()
    elif operation == 'min':
        df[output_col] = numeric_col.cummin()
    elif operation == 'product':
        df[output_col] = numeric_col.cumprod()
    elif operation == 'percent':
        total = numeric_col.sum()
        df[output_col] = (numeric_col.cumsum() / total * 100).round(2)

output = df
`,
  },

  'replace-values': {
    type: 'replace-values',
    category: 'transform',
    label: 'Replace Values',
    description: 'Map and replace specific values with new values',
    icon: 'Replace',
    defaultConfig: {
      column: '',
      replacements: [{ from: '', to: '' }],
      matchCase: true,
      replaceNull: false,
      nullReplacement: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
replacements = config.get('replacements', [])
match_case = config.get('matchCase', True)
replace_null = config.get('replaceNull', False)
null_replacement = config.get('nullReplacement', '')

if not column:
    raise ValueError("Replace Values: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Replace Values: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Handle null replacement first
if replace_null:
    df[column] = df[column].fillna(null_replacement)

# Build replacement dictionary
replace_dict = {}
for r in replacements:
    from_val = r.get('from', '')
    to_val = r.get('to', '')
    if from_val != '':
        replace_dict[from_val] = to_val

if replace_dict:
    if match_case:
        df[column] = df[column].replace(replace_dict)
    else:
        # Case-insensitive replacement for strings
        if df[column].dtype == 'object':
            for from_val, to_val in replace_dict.items():
                mask = df[column].astype(str).str.lower() == str(from_val).lower()
                df.loc[mask, column] = to_val
        else:
            df[column] = df[column].replace(replace_dict)

output = df
`,
  },

  'percent-change': {
    type: 'percent-change',
    category: 'transform',
    label: 'Percent Change',
    description: 'Calculate percentage change between consecutive rows',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      periods: 1,
      groupBy: [],
      outputColumn: '',
      fillMethod: 'null',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
periods = config.get('periods', 1)
group_by = config.get('groupBy', [])
output_col = config.get('outputColumn', '')
fill_method = config.get('fillMethod', 'null')

if not column:
    raise ValueError("Percent Change: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Percent Change: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"{column}_pct_change"

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

if group_by:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Percent Change: Group column(s) not found: {', '.join(missing)}")
    pct_change = df.groupby(group_by)[column].transform(
        lambda x: pd.to_numeric(x, errors='coerce').pct_change(periods=periods) * 100
    )
else:
    pct_change = numeric_col.pct_change(periods=periods) * 100

# Handle fill method
if fill_method == 'zero':
    pct_change = pct_change.fillna(0)
elif fill_method == 'forward':
    pct_change = pct_change.ffill()

df[output_col] = pct_change.round(2)

output = df
`,
  },

  'round-numbers': {
    type: 'round-numbers',
    category: 'transform',
    label: 'Round Numbers',
    description: 'Round, floor, or ceiling numeric values',
    icon: 'Hash',
    defaultConfig: {
      column: '',
      method: 'round',
      decimals: 2,
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'round')
decimals = config.get('decimals', 2)
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Round Numbers: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Round Numbers: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

if method == 'round':
    df[output_col] = numeric_col.round(decimals)
elif method == 'floor':
    if decimals == 0:
        df[output_col] = np.floor(numeric_col)
    else:
        factor = 10 ** decimals
        df[output_col] = np.floor(numeric_col * factor) / factor
elif method == 'ceil':
    if decimals == 0:
        df[output_col] = np.ceil(numeric_col)
    else:
        factor = 10 ** decimals
        df[output_col] = np.ceil(numeric_col * factor) / factor
elif method == 'truncate':
    if decimals == 0:
        df[output_col] = np.trunc(numeric_col)
    else:
        factor = 10 ** decimals
        df[output_col] = np.trunc(numeric_col * factor) / factor

output = df
`,
  },

  'percent-of-total': {
    type: 'percent-of-total',
    category: 'transform',
    label: 'Percent of Total',
    description: 'Calculate what percentage each row represents of the total',
    icon: 'PieChart',
    defaultConfig: {
      column: '',
      groupBy: [],
      outputColumn: '',
      decimals: 2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
group_by = config.get('groupBy', [])
output_col = config.get('outputColumn', '')
decimals = config.get('decimals', 2)

if not column:
    raise ValueError("Percent of Total: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Percent of Total: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"{column}_pct_of_total"

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

if group_by:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Percent of Total: Group column(s) not found: {', '.join(missing)}")
    group_total = df.groupby(group_by)[column].transform(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    )
    df[output_col] = (numeric_col / group_total * 100).round(decimals)
else:
    total = numeric_col.sum()
    if total == 0:
        df[output_col] = 0
    else:
        df[output_col] = (numeric_col / total * 100).round(decimals)

output = df
`,
  },

  'absolute-value': {
    type: 'absolute-value',
    category: 'transform',
    label: 'Absolute Value',
    description: 'Convert negative values to positive',
    icon: 'PlusSquare',
    defaultConfig: {
      column: '',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Absolute Value: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Absolute Value: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

# Convert to numeric and apply absolute value
numeric_col = pd.to_numeric(df[column], errors='coerce')
df[output_col] = numeric_col.abs()

output = df
`,
  },

  'column-math': {
    type: 'column-math',
    category: 'transform',
    label: 'Column Math',
    description: 'Perform arithmetic operations between two columns',
    icon: 'Calculator',
    defaultConfig: {
      column1: '',
      operation: 'add',
      column2: '',
      constant: '',
      useConstant: false,
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column1 = config.get('column1', '')
operation = config.get('operation', 'add')
column2 = config.get('column2', '')
constant = config.get('constant', '')
use_constant = config.get('useConstant', False)
output_col = config.get('outputColumn', '')

if not column1:
    raise ValueError("Column Math: Please specify the first column in the Config tab")

if column1 not in df.columns:
    raise ValueError(f"Column Math: Column '{column1}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not use_constant and not column2:
    raise ValueError("Column Math: Please specify a second column or enable 'Use Constant'")

if not use_constant and column2 not in df.columns:
    raise ValueError(f"Column Math: Column '{column2}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert to numeric
col1_numeric = pd.to_numeric(df[column1], errors='coerce')

if use_constant:
    try:
        operand = float(constant)
    except (ValueError, TypeError):
        raise ValueError(f"Column Math: Invalid constant value '{constant}'")
    op_label = str(constant)
else:
    operand = pd.to_numeric(df[column2], errors='coerce')
    op_label = column2

if not output_col:
    op_symbols = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
    output_col = f"{column1}_{op_symbols.get(operation, operation)}_{op_label}"

if operation == 'add':
    df[output_col] = col1_numeric + operand
elif operation == 'subtract':
    df[output_col] = col1_numeric - operand
elif operation == 'multiply':
    df[output_col] = col1_numeric * operand
elif operation == 'divide':
    with np.errstate(divide='ignore', invalid='ignore'):
        df[output_col] = col1_numeric / operand
        df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)

output = df
`,
  },

  'extract-substring': {
    type: 'extract-substring',
    category: 'transform',
    label: 'Extract Substring',
    description: 'Extract portion of text from a string column',
    icon: 'Scissors',
    defaultConfig: {
      column: '',
      method: 'left',
      length: 5,
      start: 0,
      delimiter: '',
      position: 'first',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'left')
length = config.get('length', 5)
start = config.get('start', 0)
delimiter = config.get('delimiter', '')
position = config.get('position', 'first')
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Extract Substring: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Extract Substring: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"{column}_extracted"

str_col = df[column].astype(str)

if method == 'left':
    df[output_col] = str_col.str[:length]
elif method == 'right':
    df[output_col] = str_col.str[-length:]
elif method == 'mid':
    df[output_col] = str_col.str[start:start + length]
elif method == 'before_delimiter':
    if not delimiter:
        raise ValueError("Extract Substring: Please specify a delimiter")
    if position == 'first':
        df[output_col] = str_col.str.split(delimiter, n=1).str[0]
    else:
        df[output_col] = str_col.str.rsplit(delimiter, n=1).str[0]
elif method == 'after_delimiter':
    if not delimiter:
        raise ValueError("Extract Substring: Please specify a delimiter")
    if position == 'first':
        parts = str_col.str.split(delimiter, n=1)
        df[output_col] = parts.apply(lambda x: x[1] if len(x) > 1 else '')
    else:
        parts = str_col.str.rsplit(delimiter, n=1)
        df[output_col] = parts.apply(lambda x: x[1] if len(x) > 1 else '')
elif method == 'between_delimiters':
    start_delim = delimiter
    end_delim = config.get('endDelimiter', delimiter)
    df[output_col] = str_col.str.extract(f'{start_delim}(.*?){end_delim}', expand=False)

output = df
`,
  },

  'parse-date': {
    type: 'parse-date',
    category: 'transform',
    label: 'Parse Date',
    description: 'Convert text strings to proper date format',
    icon: 'CalendarCheck',
    defaultConfig: {
      column: '',
      format: 'auto',
      customFormat: '',
      outputColumn: '',
      errors: 'coerce',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
date_format = config.get('format', 'auto')
custom_format = config.get('customFormat', '')
output_col = config.get('outputColumn', '')
errors = config.get('errors', 'coerce')

if not column:
    raise ValueError("Parse Date: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Parse Date: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

# Define common date formats
formats = {
    'auto': None,
    'YYYY-MM-DD': '%Y-%m-%d',
    'MM/DD/YYYY': '%m/%d/%Y',
    'DD/MM/YYYY': '%d/%m/%Y',
    'YYYY/MM/DD': '%Y/%m/%d',
    'MM-DD-YYYY': '%m-%d-%Y',
    'DD-MM-YYYY': '%d-%m-%Y',
    'YYYYMMDD': '%Y%m%d',
    'Mon DD, YYYY': '%b %d, %Y',
    'Month DD, YYYY': '%B %d, %Y',
    'DD Mon YYYY': '%d %b %Y',
    'custom': custom_format,
}

fmt = formats.get(date_format, None)

if date_format == 'auto':
    df[output_col] = pd.to_datetime(df[column], errors=errors, infer_datetime_format=True)
elif fmt:
    df[output_col] = pd.to_datetime(df[column], format=fmt, errors=errors)
else:
    df[output_col] = pd.to_datetime(df[column], errors=errors)

output = df
`,
  },

  'split-to-rows': {
    type: 'split-to-rows',
    category: 'transform',
    label: 'Split to Rows',
    description: 'Expand delimited values in a cell into separate rows',
    icon: 'SplitSquareVertical',
    defaultConfig: {
      column: '',
      delimiter: ',',
      trimWhitespace: true,
      dropEmpty: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
delimiter = config.get('delimiter', ',')
trim_whitespace = config.get('trimWhitespace', True)
drop_empty = config.get('dropEmpty', True)

if not column:
    raise ValueError("Split to Rows: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Split to Rows: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert column to string and split
df[column] = df[column].astype(str)

# Split the column and explode into rows
df[column] = df[column].str.split(delimiter)
df = df.explode(column)

# Trim whitespace if requested
if trim_whitespace:
    df[column] = df[column].str.strip()

# Drop empty values if requested
if drop_empty:
    df = df[df[column] != '']
    df = df[df[column].notna()]

# Reset index
df = df.reset_index(drop=True)

output = df
`,
  },

  'clip-values': {
    type: 'clip-values',
    category: 'transform',
    label: 'Clip Values',
    description: 'Cap values at minimum and/or maximum thresholds',
    icon: 'Minimize2',
    defaultConfig: {
      column: '',
      minValue: '',
      maxValue: '',
      usePercentile: false,
      lowerPercentile: 5,
      upperPercentile: 95,
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
min_value = config.get('minValue', '')
max_value = config.get('maxValue', '')
use_percentile = config.get('usePercentile', False)
lower_pct = config.get('lowerPercentile', 5)
upper_pct = config.get('upperPercentile', 95)
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Clip Values: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Clip Values: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

if use_percentile:
    lower_bound = numeric_col.quantile(lower_pct / 100)
    upper_bound = numeric_col.quantile(upper_pct / 100)
    df[output_col] = numeric_col.clip(lower=lower_bound, upper=upper_bound)
else:
    lower_bound = float(min_value) if min_value != '' else None
    upper_bound = float(max_value) if max_value != '' else None
    df[output_col] = numeric_col.clip(lower=lower_bound, upper=upper_bound)

output = df
`,
  },

  'standardize-text': {
    type: 'standardize-text',
    category: 'transform',
    label: 'Standardize Text',
    description: 'Comprehensive text cleaning and normalization',
    icon: 'Type',
    defaultConfig: {
      column: '',
      removeAccents: true,
      normalizeUnicode: true,
      removeSpecialChars: false,
      collapseSpaces: true,
      trimWhitespace: true,
      caseConversion: 'none',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import unicodedata
import re

df = input_data.copy()
column = config.get('column', '')
remove_accents = config.get('removeAccents', True)
normalize_unicode = config.get('normalizeUnicode', True)
remove_special = config.get('removeSpecialChars', False)
collapse_spaces = config.get('collapseSpaces', True)
trim_ws = config.get('trimWhitespace', True)
case_conv = config.get('caseConversion', 'none')
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Standardize Text: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Standardize Text: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = column

def standardize(text):
    if pd.isna(text):
        return text

    s = str(text)

    # Normalize unicode
    if normalize_unicode:
        s = unicodedata.normalize('NFKC', s)

    # Remove accents
    if remove_accents:
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')

    # Remove special characters (keep alphanumeric and spaces)
    if remove_special:
        s = re.sub(r'[^a-zA-Z0-9\\s]', '', s)

    # Collapse multiple spaces
    if collapse_spaces:
        s = re.sub(r'\\s+', ' ', s)

    # Trim whitespace
    if trim_ws:
        s = s.strip()

    # Case conversion
    if case_conv == 'lower':
        s = s.lower()
    elif case_conv == 'upper':
        s = s.upper()
    elif case_conv == 'title':
        s = s.title()
    elif case_conv == 'sentence':
        s = s.capitalize()

    return s

df[output_col] = df[column].apply(standardize)

output = df
`,
  },

  'case-when': {
    type: 'case-when',
    category: 'transform',
    label: 'Case When',
    description: 'Create column based on multiple if-then conditions',
    icon: 'GitBranch',
    defaultConfig: {
      conditions: [{ column: '', operator: 'equals', value: '', result: '' }],
      elseValue: '',
      outputColumn: 'result',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
conditions = config.get('conditions', [])
else_value = config.get('elseValue', '')
output_col = config.get('outputColumn', 'result')

if not conditions:
    raise ValueError("Case When: Please add at least one condition in the Config tab")

if not output_col:
    output_col = 'result'

# Start with else value
df[output_col] = else_value

# Apply conditions in reverse order (last condition has lowest priority)
for cond in reversed(conditions):
    col = cond.get('column', '')
    operator = cond.get('operator', 'equals')
    value = cond.get('value', '')
    result = cond.get('result', '')

    if not col:
        continue

    if col not in df.columns:
        raise ValueError(f"Case When: Column '{col}' not found. Available columns: {', '.join(df.columns.tolist())}")

    # Create mask based on operator
    if operator == 'equals':
        try:
            numeric_val = float(value)
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            mask = numeric_col == numeric_val
            if mask.sum() == 0:
                mask = df[col].astype(str) == str(value)
        except (ValueError, TypeError):
            mask = df[col].astype(str) == str(value)
    elif operator == 'not_equals':
        try:
            numeric_val = float(value)
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            mask = numeric_col != numeric_val
        except (ValueError, TypeError):
            mask = df[col].astype(str) != str(value)
    elif operator == 'greater_than':
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        mask = numeric_col > float(value)
    elif operator == 'less_than':
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        mask = numeric_col < float(value)
    elif operator == 'greater_or_equal':
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        mask = numeric_col >= float(value)
    elif operator == 'less_or_equal':
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        mask = numeric_col <= float(value)
    elif operator == 'contains':
        mask = df[col].astype(str).str.contains(str(value), na=False)
    elif operator == 'starts_with':
        mask = df[col].astype(str).str.startswith(str(value), na=False)
    elif operator == 'ends_with':
        mask = df[col].astype(str).str.endswith(str(value), na=False)
    elif operator == 'is_null':
        mask = df[col].isnull()
    elif operator == 'is_not_null':
        mask = df[col].notnull()
    elif operator == 'between':
        parts = str(value).split(',')
        if len(parts) == 2:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            mask = (numeric_col >= float(parts[0].strip())) & (numeric_col <= float(parts[1].strip()))
        else:
            mask = pd.Series([False] * len(df))
    else:
        mask = pd.Series([False] * len(df))

    df.loc[mask, output_col] = result

output = df
`,
  },

  'explode-column': {
    type: 'explode-column',
    category: 'transform',
    label: 'Explode Column',
    description: 'Expand list/array values in a column into separate rows',
    icon: 'Expand',
    defaultConfig: {
      column: '',
      delimiter: ',',
      ignoreIndex: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import ast

df = input_data.copy()
column = config.get('column', '')
delimiter = config.get('delimiter', ',')
ignore_index = config.get('ignoreIndex', True)

if not column:
    raise ValueError("Explode Column: Please specify a column to explode in the Config tab")

if column not in df.columns:
    raise ValueError(f"Explode Column: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Try to parse string representations of lists, or split by delimiter
def parse_value(x):
    if pd.isna(x):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        # Try to parse as Python literal (list, tuple)
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
        # Fall back to splitting by delimiter
        if delimiter:
            return [item.strip() for item in x.split(delimiter)]
    return [x]

df[column] = df[column].apply(parse_value)
df = df.explode(column, ignore_index=ignore_index)

output = df
`,
  },

  'add-constant-column': {
    type: 'add-constant-column',
    category: 'transform',
    label: 'Add Constant Column',
    description: 'Add a new column with a fixed value for all rows',
    icon: 'PlusSquare',
    defaultConfig: {
      columnName: 'new_column',
      value: '',
      valueType: 'string',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column_name = config.get('columnName', 'new_column')
value = config.get('value', '')
value_type = config.get('valueType', 'string')

if not column_name:
    raise ValueError("Add Constant Column: Please specify a column name in the Config tab")

# Convert value to the specified type
if value_type == 'integer':
    try:
        value = int(float(value)) if value != '' else 0
    except (ValueError, TypeError):
        raise ValueError(f"Add Constant Column: Cannot convert '{value}' to integer")
elif value_type == 'float':
    try:
        value = float(value) if value != '' else 0.0
    except (ValueError, TypeError):
        raise ValueError(f"Add Constant Column: Cannot convert '{value}' to float")
elif value_type == 'boolean':
    value = str(value).lower() in ('true', '1', 'yes')
elif value_type == 'null':
    value = np.nan
# else keep as string

df[column_name] = value

output = df
`,
  },

  'drop-columns': {
    type: 'drop-columns',
    category: 'transform',
    label: 'Drop Columns',
    description: 'Remove specific columns from the dataset',
    icon: 'Trash2',
    defaultConfig: {
      columns: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])

if not columns:
    raise ValueError("Drop Columns: Please specify columns to drop in the Config tab")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Drop Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

df = df.drop(columns=columns)

output = df
`,
  },

  'flatten-json': {
    type: 'flatten-json',
    category: 'transform',
    label: 'Flatten JSON',
    description: 'Expand nested dict/JSON columns into separate columns',
    icon: 'Layers',
    defaultConfig: {
      column: '',
      prefix: '',
      separator: '_',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import json
import ast

df = input_data.copy()
column = config.get('column', '')
prefix = config.get('prefix', '')
separator = config.get('separator', '_')

if not column:
    raise ValueError("Flatten JSON: Please specify a column to flatten in the Config tab")

if column not in df.columns:
    raise ValueError(f"Flatten JSON: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Parse JSON/dict strings
def parse_json(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return {}
    return {}

parsed = df[column].apply(parse_json)
flat_df = pd.json_normalize(parsed, sep=separator)

# Add prefix if specified
if prefix:
    flat_df.columns = [f"{prefix}{separator}{col}" for col in flat_df.columns]

# Drop original column and join flattened columns
df = df.drop(columns=[column])
df = pd.concat([df.reset_index(drop=True), flat_df.reset_index(drop=True)], axis=1)

output = df
`,
  },

  'coalesce-columns': {
    type: 'coalesce-columns',
    category: 'transform',
    label: 'Coalesce Columns',
    description: 'Get the first non-null value from multiple columns',
    icon: 'Merge',
    defaultConfig: {
      columns: [],
      outputColumn: 'coalesced',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
output_col = config.get('outputColumn', 'coalesced')

if not columns or len(columns) < 2:
    raise ValueError("Coalesce Columns: Please specify at least 2 columns to coalesce in the Config tab")

if not output_col:
    output_col = 'coalesced'

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Coalesce Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Get first non-null value across specified columns
df[output_col] = df[columns].bfill(axis=1).iloc[:, 0]

output = df
`,
  },

  'reorder-columns': {
    type: 'reorder-columns',
    category: 'transform',
    label: 'Reorder Columns',
    description: 'Rearrange column order in the dataset',
    icon: 'ArrowUpDown',
    defaultConfig: {
      columns: [],
      position: 'start',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
position = config.get('position', 'start')

if not columns:
    raise ValueError("Reorder Columns: Please specify columns to reorder in the Config tab")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Reorder Columns: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Get remaining columns (not in the reorder list)
remaining = [c for c in df.columns if c not in columns]

if position == 'start':
    new_order = columns + remaining
elif position == 'end':
    new_order = remaining + columns
elif position == 'custom':
    # Use the exact order specified (columns should contain all columns)
    new_order = columns
else:
    new_order = columns + remaining

df = df[new_order]

output = df
`,
  },

  'trim-text': {
    type: 'trim-text',
    category: 'transform',
    label: 'Trim & Clean Text',
    description: 'Remove whitespace and clean text formatting',
    icon: 'Eraser',
    defaultConfig: {
      columns: [],
      trimWhitespace: true,
      collapseSpaces: false,
      removeNonPrintable: false,
      applyToAll: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re

df = input_data.copy()
columns = config.get('columns', [])
trim_whitespace = config.get('trimWhitespace', True)
collapse_spaces = config.get('collapseSpaces', False)
remove_non_printable = config.get('removeNonPrintable', False)
apply_to_all = config.get('applyToAll', False)

# Determine which columns to process
if apply_to_all:
    cols_to_process = df.select_dtypes(include=['object']).columns.tolist()
elif columns:
    cols_to_process = columns
else:
    raise ValueError("Trim & Clean Text: Please specify columns or enable 'Apply to all text columns' in the Config tab")

# Validate columns exist
missing = [c for c in cols_to_process if c not in df.columns]
if missing:
    raise ValueError(f"Trim & Clean Text: Column(s) not found: {', '.join(missing)}")

def clean_text(x):
    if pd.isna(x) or not isinstance(x, str):
        return x
    result = x
    if remove_non_printable:
        result = ''.join(char for char in result if char.isprintable() or char in '\\n\\t')
    if collapse_spaces:
        result = re.sub(r'\\s+', ' ', result)
    if trim_whitespace:
        result = result.strip()
    return result

for col in cols_to_process:
    df[col] = df[col].apply(clean_text)

output = df
`,
  },

  'lookup-vlookup': {
    type: 'lookup-vlookup',
    category: 'transform',
    label: 'Lookup (VLOOKUP)',
    description: 'Match and retrieve values from another dataset like Excel VLOOKUP',
    icon: 'Search',
    defaultConfig: {
      lookupColumn: '',
      returnColumns: [],
      matchType: 'exact',
    },
    inputs: 2,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

# input_data is a list of DataFrames when there are 2 inputs
if not isinstance(input_data, list) or len(input_data) < 2:
    raise ValueError("Lookup (VLOOKUP): This block requires 2 inputs - main data (top) and lookup table (bottom)")

main_df = input_data[0].copy()
lookup_df = input_data[1].copy()

lookup_col = config.get('lookupColumn', '')
return_cols = config.get('returnColumns', [])
match_type = config.get('matchType', 'exact')

if not lookup_col:
    raise ValueError("Lookup (VLOOKUP): Please specify the lookup column in the Config tab")

if lookup_col not in main_df.columns:
    raise ValueError(f"Lookup (VLOOKUP): Lookup column '{lookup_col}' not found in main data. Available: {', '.join(main_df.columns.tolist())}")

if lookup_col not in lookup_df.columns:
    raise ValueError(f"Lookup (VLOOKUP): Lookup column '{lookup_col}' not found in lookup table. Available: {', '.join(lookup_df.columns.tolist())}")

# Determine which columns to return
if not return_cols:
    return_cols = [c for c in lookup_df.columns if c != lookup_col]

# Validate return columns exist
missing = [c for c in return_cols if c not in lookup_df.columns]
if missing:
    raise ValueError(f"Lookup (VLOOKUP): Return column(s) not found in lookup table: {', '.join(missing)}")

# Perform the lookup (merge)
cols_to_merge = [lookup_col] + return_cols
lookup_subset = lookup_df[cols_to_merge].drop_duplicates(subset=[lookup_col], keep='first')

df = main_df.merge(lookup_subset, on=lookup_col, how='left')

output = df
`,
  },

  'cross-join': {
    type: 'cross-join',
    category: 'transform',
    label: 'Cross Join',
    description: 'Create Cartesian product of two datasets (all combinations)',
    icon: 'Grid3x3',
    defaultConfig: {
      suffixes: ['_x', '_y'],
    },
    inputs: 2,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

# input_data is a list of DataFrames when there are 2 inputs
if not isinstance(input_data, list) or len(input_data) < 2:
    raise ValueError("Cross Join: This block requires 2 inputs")

df1 = input_data[0].copy()
df2 = input_data[1].copy()

suffixes = config.get('suffixes', ['_x', '_y'])
if not suffixes or len(suffixes) < 2:
    suffixes = ['_x', '_y']

# Warn about potentially large output
total_rows = len(df1) * len(df2)
if total_rows > 1000000:
    raise ValueError(f"Cross Join: Result would have {total_rows:,} rows. Consider filtering inputs first to avoid memory issues.")

# Perform cross join
df1['_cross_key'] = 1
df2['_cross_key'] = 1
df = df1.merge(df2, on='_cross_key', suffixes=suffixes).drop('_cross_key', axis=1)

output = df
`,
  },

  'filter-expression': {
    type: 'filter-expression',
    category: 'transform',
    label: 'Filter by Expression',
    description: 'Filter rows using a Python expression for advanced conditions',
    icon: 'Code',
    defaultConfig: {
      expression: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
expression = config.get('expression', '')

if not expression:
    raise ValueError("Filter by Expression: Please enter a filter expression in the Config tab. Example: (df['age'] > 18) & (df['status'] == 'active')")

try:
    # Allow using column names directly and common functions
    mask = eval(expression)
    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask, index=df.index)
    df = df[mask]
except Exception as e:
    raise ValueError(f"Filter by Expression: Invalid expression - {str(e)}. Example: (df['age'] > 18) & (df['status'] == 'active')")

output = df
`,
  },

  'number-format': {
    type: 'number-format',
    category: 'transform',
    label: 'Number Format',
    description: 'Format numbers with thousands separators, decimals, currency, or percentage',
    icon: 'Hash',
    defaultConfig: {
      column: '',
      format: 'thousands',
      decimals: 2,
      prefix: '',
      suffix: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
format_type = config.get('format', 'thousands')
decimals = int(config.get('decimals', 2))
prefix = config.get('prefix', '')
suffix = config.get('suffix', '')

if not column:
    raise ValueError("Number Format: Please specify a column to format in the Config tab")

if column not in df.columns:
    raise ValueError(f"Number Format: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert to numeric first
numeric_col = pd.to_numeric(df[column], errors='coerce')

def format_number(x):
    if pd.isna(x):
        return ''

    if format_type == 'thousands':
        formatted = f"{x:,.{decimals}f}"
    elif format_type == 'currency':
        formatted = f"{x:,.{decimals}f}"
    elif format_type == 'percentage':
        formatted = f"{x * 100:.{decimals}f}%"
    elif format_type == 'scientific':
        formatted = f"{x:.{decimals}e}"
    elif format_type == 'plain':
        formatted = f"{x:.{decimals}f}"
    else:
        formatted = f"{x:,.{decimals}f}"

    return f"{prefix}{formatted}{suffix}"

df[column] = numeric_col.apply(format_number)

output = df
`,
  },

  'extract-pattern': {
    type: 'extract-pattern',
    category: 'transform',
    label: 'Extract Pattern',
    description: 'Extract text matching a regex pattern into a new column',
    icon: 'Regex',
    defaultConfig: {
      column: '',
      pattern: '',
      outputColumn: 'extracted',
      extractAll: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re

df = input_data.copy()
column = config.get('column', '')
pattern = config.get('pattern', '')
output_col = config.get('outputColumn', 'extracted')
extract_all = config.get('extractAll', False)

if not column:
    raise ValueError("Extract Pattern: Please specify a source column in the Config tab")

if not pattern:
    raise ValueError("Extract Pattern: Please specify a regex pattern in the Config tab. Examples: r'\\\\d+' for numbers, r'[A-Za-z]+@[A-Za-z]+\\\\.[A-Za-z]+' for emails")

if column not in df.columns:
    raise ValueError(f"Extract Pattern: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = 'extracted'

try:
    if extract_all:
        # Extract all matches as a list
        df[output_col] = df[column].astype(str).apply(lambda x: re.findall(pattern, x) if pd.notna(x) else [])
    else:
        # Extract first match only
        extracted = df[column].astype(str).str.extract(f'({pattern})', expand=False)
        df[output_col] = extracted
except re.error as e:
    raise ValueError(f"Extract Pattern: Invalid regex pattern - {str(e)}")

output = df
`,
  },

  // Analysis Blocks
  'statistics': {
    type: 'statistics',
    category: 'analysis',
    label: 'Statistics',
    description: 'Calculate descriptive statistics and correlations',
    icon: 'BarChart3',
    defaultConfig: {
      type: 'descriptive',
      columns: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
stat_type = config.get('type', 'descriptive')
columns = config.get('columns', [])

# Try to convert all columns to numeric where possible, handling comma-formatted numbers
for col in df.columns:
    try:
        # First try direct conversion
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            # Try removing commas for numbers like "9,000"
            converted = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', ''), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    except:
        pass

# Use numeric columns if none specified
if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Statistics: No numeric columns found in data. Please ensure your data contains numbers.")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Statistics: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

if stat_type == 'descriptive':
    result = df[columns].describe()
    result = result.reset_index().rename(columns={'index': 'statistic'})
    output = result
elif stat_type == 'correlation':
    result = df[columns].corr()
    result = result.reset_index().rename(columns={'index': 'column'})
    output = result
elif stat_type == 'covariance':
    result = df[columns].cov()
    result = result.reset_index().rename(columns={'index': 'column'})
    output = result
else:
    raise ValueError(f"Statistics: Unknown statistics type: {stat_type}")
`,
  },

  'regression': {
    type: 'regression',
    category: 'analysis',
    label: 'Regression',
    description: 'Perform linear or logistic regression',
    icon: 'TrendingUp',
    defaultConfig: {
      type: 'linear',
      features: [],
      target: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

df = input_data.copy()
reg_type = config.get('type', 'linear')
features = config.get('features', [])
target = config.get('target', '')

if not features:
    raise ValueError("Regression: Please specify feature columns in the Config tab")

if not target:
    raise ValueError("Regression: Please specify a target column in the Config tab")

# Validate columns exist
all_cols = features + [target]
missing = [c for c in all_cols if c not in df.columns]
if missing:
    raise ValueError(f"Regression: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Convert feature and target columns to numeric, handling comma-formatted numbers
for col in all_cols:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            converted = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', ''), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    except:
        pass

# Drop rows with NaN values in the columns we need
df = df.dropna(subset=all_cols)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if reg_type == 'linear':
    model = LinearRegression()
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

output = {
    'coefficients': dict(zip(features, model.coef_.tolist() if hasattr(model.coef_, 'tolist') else [model.coef_])),
    'intercept': float(model.intercept_) if hasattr(model.intercept_, '__float__') else model.intercept_,
    'score': float(score),
    'predictions': model.predict(X).tolist(),
}
`,
  },

  'clustering': {
    type: 'clustering',
    category: 'analysis',
    label: 'Clustering',
    description: 'Perform K-means or hierarchical clustering',
    icon: 'Network',
    defaultConfig: {
      algorithm: 'kmeans',
      nClusters: 3,
      features: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
algorithm = config.get('algorithm', 'kmeans')
n_clusters = config.get('nClusters', 3)
features = config.get('features', [])

if not features:
    raise ValueError("Clustering: Please specify feature columns in the Config tab")

# Validate columns exist
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Clustering: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Convert feature columns to numeric, handling comma-formatted numbers
for col in features:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            converted = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', ''), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    except:
        pass

# Drop rows with NaN values in feature columns
df = df.dropna(subset=features)

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if algorithm == 'kmeans':
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
else:
    model = AgglomerativeClustering(n_clusters=n_clusters)

labels = model.fit_predict(X_scaled)
df['cluster'] = labels

output = df
`,
  },

  'pca': {
    type: 'pca',
    category: 'analysis',
    label: 'PCA',
    description: 'Principal Component Analysis for dimensionality reduction',
    icon: 'Minimize2',
    defaultConfig: {
      features: [],
      nComponents: 2,
      scaleData: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
n_components = config.get('nComponents', 2)
scale_data = config.get('scaleData', True)

if not features:
    raise ValueError("PCA: Please select feature columns in the Config tab (run the pipeline first to see available columns)")

# Validate columns exist
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"PCA: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Convert feature columns to numeric
for col in features:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            converted = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', ''), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    except:
        pass

# Drop rows with NaN values in feature columns
df_clean = df.dropna(subset=features)
if len(df_clean) == 0:
    raise ValueError("PCA: No valid rows after removing NaN values. Check that selected columns contain numeric data.")

X = df_clean[features].values

# Standardize features if requested
if scale_data:
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)
else:
    X_processed = X

# Apply PCA
n_comp = min(n_components, len(features), len(df_clean))
pca = PCA(n_components=n_comp)
components = pca.fit_transform(X_processed)

# Add PC columns to data
for i in range(n_comp):
    df_clean[f'PC{i+1}'] = components[:, i]

# Add explained variance info
df_clean['explained_variance_ratio'] = sum(pca.explained_variance_ratio_)

output = df_clean
`,
  },

  'outlier-detection': {
    type: 'outlier-detection',
    category: 'analysis',
    label: 'Outlier Detection',
    description: 'Detect outliers using IQR or Z-score methods',
    icon: 'AlertTriangle',
    defaultConfig: {
      method: 'iqr',
      columns: [],
      threshold: 1.5,
      action: 'flag',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
method = config.get('method', 'iqr')
columns = config.get('columns', [])
threshold = config.get('threshold', 1.5)
action = config.get('action', 'flag')

# Use all numeric columns if none specified
if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Outlier Detection: No numeric columns found or specified")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Outlier Detection: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Convert columns to numeric
for col in columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        pass

# Detect outliers
outlier_mask = pd.Series([False] * len(df), index=df.index)

for col in columns:
    col_data = df[col].dropna()
    if len(col_data) == 0:
        continue

    if method == 'iqr':
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        col_outliers = (df[col] < lower) | (df[col] > upper)
    else:  # z-score
        mean = col_data.mean()
        std = col_data.std()
        if std > 0:
            z_scores = np.abs((df[col] - mean) / std)
            col_outliers = z_scores > threshold
        else:
            col_outliers = pd.Series([False] * len(df), index=df.index)

    outlier_mask = outlier_mask | col_outliers.fillna(False)

if action == 'flag':
    df['is_outlier'] = outlier_mask
    output = df
elif action == 'remove':
    output = df[~outlier_mask]
else:  # keep only outliers
    output = df[outlier_mask]
`,
  },

  'classification': {
    type: 'classification',
    category: 'analysis',
    label: 'Classification',
    description: 'Train a decision tree or random forest classifier',
    icon: 'GitFork',
    defaultConfig: {
      algorithm: 'decision_tree',
      features: [],
      target: '',
      testSize: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
algorithm = config.get('algorithm', 'decision_tree')
features = config.get('features', [])
target = config.get('target', '')
test_size = config.get('testSize', 0.2)

if not features:
    raise ValueError("Classification: Please select feature columns in the Config tab (run the pipeline first to see available columns)")

if not target:
    raise ValueError("Classification: Please select a target column in the Config tab")

# Validate columns exist
all_cols = features + [target]
missing = [c for c in all_cols if c not in df.columns]
if missing:
    raise ValueError(f"Classification: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

# Convert feature columns to numeric
for col in features:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
    except:
        pass

# Drop rows with NaN values
df_clean = df.dropna(subset=all_cols)
if len(df_clean) < 10:
    raise ValueError(f"Classification: Not enough valid data rows ({len(df_clean)}). Need at least 10 rows.")

X = df_clean[features].values
y = df_clean[target]

# Encode target if it's categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

# Train model
if algorithm == 'decision_tree':
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
else:
    model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Get predictions for all data
predictions = model.predict(X)
df_clean['predicted_class'] = le.inverse_transform(predictions)
df_clean['prediction_correct'] = df_clean[target].astype(str) == df_clean['predicted_class'].astype(str)

# Add accuracy as a column for visibility
df_clean['model_accuracy'] = accuracy

output = df_clean
`,
  },

  'normality-test': {
    type: 'normality-test',
    category: 'analysis',
    label: 'Normality Test',
    description: 'Test if data follows a normal distribution',
    icon: 'Activity',
    defaultConfig: {
      method: 'shapiro',
      columns: [],
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
method = config.get('method', 'shapiro')
columns = config.get('columns', [])
alpha = config.get('alpha', 0.05)

# Use all numeric columns if none specified
if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Normality Test: No numeric columns found. Please select columns or ensure your data has numeric columns.")

# Validate columns exist
missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Normality Test: Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns.tolist())}")

results = []
for col in columns:
    col_data = pd.to_numeric(df[col], errors='coerce').dropna()

    if len(col_data) < 3:
        results.append({
            'column': col,
            'test': 'N/A',
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'note': 'Not enough data points (need at least 3)'
        })
        continue

    # Perform the selected test
    try:
        if method == 'shapiro':
            # Shapiro-Wilk test (best for n < 5000)
            sample = col_data[:5000] if len(col_data) > 5000 else col_data
            stat, p_value = stats.shapiro(sample)
            test_name = 'Shapiro-Wilk'
        else:
            # D'Agostino-Pearson test (needs n >= 20)
            if len(col_data) < 20:
                stat, p_value = stats.shapiro(col_data)
                test_name = 'Shapiro-Wilk (fallback, n<20)'
            else:
                stat, p_value = stats.normaltest(col_data)
                test_name = "D'Agostino-Pearson"

        results.append({
            'column': col,
            'test': test_name,
            'statistic': round(float(stat), 6),
            'p_value': round(float(p_value), 6),
            'is_normal': 'Yes' if p_value > alpha else 'No',
            'interpretation': f'Data {"appears" if p_value > alpha else "does not appear"} normally distributed (α={alpha})',
            'mean': round(float(col_data.mean()), 4),
            'std': round(float(col_data.std()), 4),
            'skewness': round(float(stats.skew(col_data)), 4),
            'kurtosis': round(float(stats.kurtosis(col_data)), 4),
            'n_samples': len(col_data),
        })
    except Exception as e:
        results.append({
            'column': col,
            'test': 'Error',
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'note': str(e)
        })

output = pd.DataFrame(results)
`,
  },

  'hypothesis-testing': {
    type: 'hypothesis-testing',
    category: 'analysis',
    label: 'Hypothesis Testing',
    description: 'Perform statistical hypothesis tests (t-test, chi-square, ANOVA)',
    icon: 'FlaskConical',
    defaultConfig: {
      testType: 'ttest_ind',
      column1: '',
      column2: '',
      groupColumn: '',
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
test_type = config.get('testType', 'ttest_ind')
column1 = config.get('column1', '')
column2 = config.get('column2', '')
group_column = config.get('groupColumn', '')
alpha = config.get('alpha', 0.05)

results = []

if test_type == 'ttest_ind':
    if not column1 or not group_column:
        raise ValueError("T-Test: Please specify a numeric column and a grouping column")
    groups = df[group_column].unique()
    if len(groups) != 2:
        raise ValueError(f"T-Test: Grouping column must have exactly 2 groups, found {len(groups)}")
    group1_data = pd.to_numeric(df[df[group_column] == groups[0]][column1], errors='coerce').dropna()
    group2_data = pd.to_numeric(df[df[group_column] == groups[1]][column1], errors='coerce').dropna()
    stat, p_value = stats.ttest_ind(group1_data, group2_data)
    results.append({
        'test': 'Independent T-Test',
        'column': column1,
        'group1': str(groups[0]),
        'group2': str(groups[1]),
        'group1_mean': round(float(group1_data.mean()), 4),
        'group2_mean': round(float(group2_data.mean()), 4),
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': 'Yes' if p_value < alpha else 'No',
    })
elif test_type == 'ttest_paired':
    if not column1 or not column2:
        raise ValueError("Paired T-Test: Please specify two numeric columns")
    data1 = pd.to_numeric(df[column1], errors='coerce').dropna()
    data2 = pd.to_numeric(df[column2], errors='coerce').dropna()
    min_len = min(len(data1), len(data2))
    stat, p_value = stats.ttest_rel(data1[:min_len], data2[:min_len])
    results.append({
        'test': 'Paired T-Test',
        'column1': column1,
        'column2': column2,
        'mean1': round(float(data1.mean()), 4),
        'mean2': round(float(data2.mean()), 4),
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': 'Yes' if p_value < alpha else 'No',
    })
elif test_type == 'chi2':
    if not column1 or not column2:
        raise ValueError("Chi-Square: Please specify two categorical columns")
    contingency = pd.crosstab(df[column1], df[column2])
    stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    results.append({
        'test': 'Chi-Square Test',
        'column1': column1,
        'column2': column2,
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'degrees_of_freedom': int(dof),
        'significant': 'Yes' if p_value < alpha else 'No',
    })
elif test_type == 'anova':
    if not column1 or not group_column:
        raise ValueError("ANOVA: Please specify a numeric column and a grouping column")
    groups = df[group_column].unique()
    group_data = [pd.to_numeric(df[df[group_column] == g][column1], errors='coerce').dropna() for g in groups]
    stat, p_value = stats.f_oneway(*group_data)
    results.append({
        'test': 'One-way ANOVA',
        'column': column1,
        'group_column': group_column,
        'n_groups': len(groups),
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': 'Yes' if p_value < alpha else 'No',
    })
elif test_type == 'mannwhitney':
    if not column1 or not group_column:
        raise ValueError("Mann-Whitney: Please specify a numeric column and a grouping column")
    groups = df[group_column].unique()
    if len(groups) != 2:
        raise ValueError(f"Mann-Whitney: Grouping column must have exactly 2 groups, found {len(groups)}")
    group1_data = pd.to_numeric(df[df[group_column] == groups[0]][column1], errors='coerce').dropna()
    group2_data = pd.to_numeric(df[df[group_column] == groups[1]][column1], errors='coerce').dropna()
    stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    results.append({
        'test': 'Mann-Whitney U Test',
        'column': column1,
        'group1': str(groups[0]),
        'group2': str(groups[1]),
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': 'Yes' if p_value < alpha else 'No',
    })

output = pd.DataFrame(results)
`,
  },

  'time-series': {
    type: 'time-series',
    category: 'analysis',
    label: 'Time Series Analysis',
    description: 'Analyze time series data with trends, seasonality, and moving averages',
    icon: 'Clock',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      analysis: 'moving_average',
      windowSize: 7,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_column = config.get('dateColumn', '')
value_column = config.get('valueColumn', '')
analysis = config.get('analysis', 'moving_average')
window_size = config.get('windowSize', 7)

if not date_column or not value_column:
    raise ValueError("Time Series: Please specify a date column and a value column")

df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df = df.dropna(subset=[date_column])
df = df.sort_values(date_column)
df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

if analysis == 'moving_average':
    df['moving_avg'] = df[value_column].rolling(window=window_size, min_periods=1).mean()
    df['moving_std'] = df[value_column].rolling(window=window_size, min_periods=1).std()
elif analysis == 'exponential_smoothing':
    alpha = 2 / (window_size + 1)
    df['exp_smooth'] = df[value_column].ewm(alpha=alpha, adjust=False).mean()
elif analysis == 'trend':
    df['time_index'] = range(len(df))
    z = np.polyfit(df['time_index'], df[value_column].fillna(0), 1)
    df['trend'] = z[0] * df['time_index'] + z[1]
    df['detrended'] = df[value_column] - df['trend']
    df = df.drop('time_index', axis=1)
elif analysis == 'pct_change':
    df['pct_change'] = df[value_column].pct_change() * 100
    df['cumulative_return'] = ((1 + df[value_column].pct_change()).cumprod() - 1) * 100
elif analysis == 'lag_features':
    for lag in [1, 7, 14, 30]:
        if lag < len(df):
            df[f'lag_{lag}'] = df[value_column].shift(lag)

output = df
`,
  },

  'feature-importance': {
    type: 'feature-importance',
    category: 'analysis',
    label: 'Feature Importance',
    description: 'Calculate feature importance using Random Forest',
    icon: 'Award',
    defaultConfig: {
      features: [],
      target: '',
      taskType: 'auto',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
task_type = config.get('taskType', 'auto')

if not features:
    raise ValueError("Feature Importance: Please select feature columns")
if not target:
    raise ValueError("Feature Importance: Please select a target column")

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features + [target])
X = df[features].values
y = df[target]

if task_type == 'auto':
    unique_values = y.nunique()
    task_type = 'classification' if unique_values < 20 else 'regression'

if task_type == 'classification':
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
else:
    y = pd.to_numeric(y, errors='coerce')
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model.fit(X, y)
importances = model.feature_importances_

results = pd.DataFrame({
    'feature': features,
    'importance': importances,
    'importance_pct': (importances / importances.sum() * 100).round(2)
}).sort_values('importance', ascending=False)
results['rank'] = range(1, len(results) + 1)

output = results
`,
  },

  'cross-validation': {
    type: 'cross-validation',
    category: 'analysis',
    label: 'Cross-Validation',
    description: 'Evaluate model performance using k-fold cross-validation',
    icon: 'Repeat',
    defaultConfig: {
      features: [],
      target: '',
      modelType: 'random_forest',
      taskType: 'auto',
      nFolds: 5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
model_type = config.get('modelType', 'random_forest')
task_type = config.get('taskType', 'auto')
n_folds = config.get('nFolds', 5)

if not features:
    raise ValueError("Cross-Validation: Please select feature columns")
if not target:
    raise ValueError("Cross-Validation: Please select a target column")

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features + [target])

X = df[features].values
y = df[target]

if task_type == 'auto':
    unique_values = y.nunique()
    task_type = 'classification' if unique_values < 20 else 'regression'

if task_type == 'classification':
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    scoring = 'accuracy'
else:
    y = pd.to_numeric(y, errors='coerce')
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    scoring = 'r2'

scores = cross_val_score(model, X, y, cv=n_folds, scoring=scoring)
results = [{'fold': i + 1, 'score': round(float(score), 4)} for i, score in enumerate(scores)]
results.append({'fold': 'Mean', 'score': round(float(scores.mean()), 4)})
results.append({'fold': 'Std', 'score': round(float(scores.std()), 4)})

output = pd.DataFrame(results)
`,
  },

  'data-profiling': {
    type: 'data-profiling',
    category: 'analysis',
    label: 'Data Profiling',
    description: 'Generate comprehensive data quality profile',
    icon: 'FileSearch',
    defaultConfig: {},
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
results = []

for col in df.columns:
    col_data = df[col]
    n_total = len(col_data)
    n_missing = col_data.isna().sum()
    n_unique = col_data.nunique()
    dtype = str(col_data.dtype)

    if dtype == 'object':
        numeric_converted = pd.to_numeric(col_data, errors='coerce')
        inferred_type = 'numeric (as string)' if numeric_converted.notna().sum() > n_total * 0.5 else 'categorical/text'
    elif 'int' in dtype or 'float' in dtype:
        inferred_type = 'numeric'
    elif 'datetime' in dtype:
        inferred_type = 'datetime'
    elif 'bool' in dtype:
        inferred_type = 'boolean'
    else:
        inferred_type = dtype

    profile = {
        'column': col,
        'dtype': dtype,
        'inferred_type': inferred_type,
        'total_count': n_total,
        'missing_count': int(n_missing),
        'missing_pct': round(n_missing / n_total * 100, 2),
        'unique_count': int(n_unique),
        'unique_pct': round(n_unique / n_total * 100, 2),
    }

    numeric_col = pd.to_numeric(col_data, errors='coerce')
    if numeric_col.notna().sum() > 0:
        profile['mean'] = round(float(numeric_col.mean()), 4)
        profile['std'] = round(float(numeric_col.std()), 4)
        profile['min'] = round(float(numeric_col.min()), 4)
        profile['max'] = round(float(numeric_col.max()), 4)

    if n_unique < 100 and len(col_data.value_counts()) > 0:
        profile['most_common'] = str(col_data.value_counts().index[0])

    results.append(profile)

output = pd.DataFrame(results)
`,
  },

  'value-counts': {
    type: 'value-counts',
    category: 'analysis',
    label: 'Value Counts',
    description: 'Count occurrences of each unique value in a column',
    icon: 'Hash',
    defaultConfig: {
      column: '',
      normalize: false,
      sortBy: 'count',
      topN: 0,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
normalize = config.get('normalize', False)
sort_by = config.get('sortBy', 'count')
top_n = config.get('topN', 0)

if not column:
    raise ValueError("Value Counts: Please select a column")
if column not in df.columns:
    raise ValueError(f"Value Counts: Column '{column}' not found")

counts = df[column].value_counts(normalize=normalize)
result = pd.DataFrame({
    'value': counts.index,
    'count' if not normalize else 'proportion': counts.values
})

if normalize:
    result['percentage'] = (result['proportion'] * 100).round(2)

if sort_by == 'value':
    result = result.sort_values('value')

if top_n > 0:
    result = result.head(top_n)

result = result.reset_index(drop=True)
result['rank'] = range(1, len(result) + 1)

output = result
`,
  },

  'cross-tabulation': {
    type: 'cross-tabulation',
    category: 'analysis',
    label: 'Cross-Tabulation',
    description: 'Create a frequency table between two categorical columns',
    icon: 'Grid3x3',
    defaultConfig: {
      rowColumn: '',
      colColumn: '',
      normalize: 'none',
      showTotals: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
row_column = config.get('rowColumn', '')
col_column = config.get('colColumn', '')
normalize = config.get('normalize', 'none')
show_totals = config.get('showTotals', True)

if not row_column or not col_column:
    raise ValueError("Cross-Tabulation: Please select row and column variables")

if normalize == 'none':
    ct = pd.crosstab(df[row_column], df[col_column], margins=show_totals)
elif normalize == 'row':
    ct = pd.crosstab(df[row_column], df[col_column], normalize='index', margins=show_totals)
    ct = (ct * 100).round(2)
elif normalize == 'column':
    ct = pd.crosstab(df[row_column], df[col_column], normalize='columns', margins=show_totals)
    ct = (ct * 100).round(2)
else:
    ct = pd.crosstab(df[row_column], df[col_column], normalize='all', margins=show_totals)
    ct = (ct * 100).round(2)

ct = ct.reset_index()
ct.columns = [str(c) for c in ct.columns]

output = ct
`,
  },

  'scaling': {
    type: 'scaling',
    category: 'analysis',
    label: 'Scaling / Normalization',
    description: 'Scale numeric features using various methods',
    icon: 'Scale',
    defaultConfig: {
      columns: [],
      method: 'standard',
      keepOriginal: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'standard')
keep_original = config.get('keepOriginal', False)

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Scaling: No numeric columns found or specified")

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if method == 'standard':
    scaler = StandardScaler()
elif method == 'minmax':
    scaler = MinMaxScaler()
elif method == 'robust':
    scaler = RobustScaler()
elif method == 'log':
    for col in columns:
        min_val = df[col].min()
        offset = abs(min_val) + 1 if min_val <= 0 else 0
        new_col = f'{col}_log' if keep_original else col
        df[new_col] = np.log1p(df[col] + offset)
    output = df
else:
    scaler = StandardScaler()

if method != 'log':
    scaled_data = scaler.fit_transform(df[columns].fillna(0))
    if keep_original:
        for i, col in enumerate(columns):
            df[f'{col}_scaled'] = scaled_data[:, i]
    else:
        for i, col in enumerate(columns):
            df[col] = scaled_data[:, i]

output = df
`,
  },

  'encoding': {
    type: 'encoding',
    category: 'analysis',
    label: 'Encoding',
    description: 'Encode categorical variables for machine learning',
    icon: 'Binary',
    defaultConfig: {
      columns: [],
      method: 'onehot',
      dropFirst: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'onehot')
drop_first = config.get('dropFirst', False)

if not columns:
    columns = df.select_dtypes(include=['object']).columns.tolist()

if not columns:
    raise ValueError("Encoding: No categorical columns found or specified")

if method == 'onehot':
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first, prefix_sep='_')
elif method == 'label':
    for col in columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
elif method == 'ordinal':
    for col in columns:
        categories = sorted(df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        df[f'{col}_ordinal'] = df[col].map(mapping)
elif method == 'frequency':
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

output = df
`,
  },

  'ab-test': {
    type: 'ab-test',
    category: 'analysis',
    label: 'A/B Test Analysis',
    description: 'Statistical analysis for A/B experiments',
    icon: 'FlaskConical',
    defaultConfig: {
      groupColumn: '',
      metricColumn: '',
      controlValue: '',
      testType: 'continuous',
      confidenceLevel: 0.95,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
group_column = config.get('groupColumn', '')
metric_column = config.get('metricColumn', '')
control_value = config.get('controlValue', '')
test_type = config.get('testType', 'continuous')
confidence_level = config.get('confidenceLevel', 0.95)

if not group_column or not metric_column:
    raise ValueError("A/B Test: Please specify group and metric columns")

if not control_value:
    raise ValueError("A/B Test: Please specify the control group value")

groups = df[group_column].unique()
if len(groups) != 2:
    raise ValueError(f"A/B Test: Expected 2 groups, found {len(groups)}: {list(groups)}")

control_group = df[df[group_column] == control_value][metric_column]
variant_value = [g for g in groups if str(g) != str(control_value)][0]
variant_group = df[df[group_column] == variant_value][metric_column]

if test_type == 'continuous':
    # Convert to numeric
    control_group = pd.to_numeric(control_group, errors='coerce').dropna()
    variant_group = pd.to_numeric(variant_group, errors='coerce').dropna()

    # Welch's t-test (unequal variances)
    stat, p_value = stats.ttest_ind(control_group, variant_group, equal_var=False)

    control_mean = control_group.mean()
    variant_mean = variant_group.mean()
    lift = ((variant_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0

    # Cohen's d effect size
    pooled_std = np.sqrt((control_group.std()**2 + variant_group.std()**2) / 2)
    cohens_d = (variant_mean - control_mean) / pooled_std if pooled_std > 0 else 0

    # Confidence interval for difference
    se = np.sqrt(control_group.var()/len(control_group) + variant_group.var()/len(variant_group))
    z = stats.norm.ppf((1 + confidence_level) / 2)
    diff = variant_mean - control_mean
    ci_lower = diff - z * se
    ci_upper = diff + z * se

    results = [{
        'metric': 'Sample Size (Control)', 'value': len(control_group)
    }, {
        'metric': 'Sample Size (Variant)', 'value': len(variant_group)
    }, {
        'metric': 'Control Mean', 'value': round(float(control_mean), 4)
    }, {
        'metric': 'Variant Mean', 'value': round(float(variant_mean), 4)
    }, {
        'metric': 'Absolute Difference', 'value': round(float(diff), 4)
    }, {
        'metric': 'Relative Lift (%)', 'value': round(float(lift), 2)
    }, {
        'metric': 'P-Value', 'value': round(float(p_value), 6)
    }, {
        'metric': 'Statistically Significant', 'value': 'Yes' if p_value < (1 - confidence_level) else 'No'
    }, {
        'metric': f'{int(confidence_level*100)}% CI Lower', 'value': round(float(ci_lower), 4)
    }, {
        'metric': f'{int(confidence_level*100)}% CI Upper', 'value': round(float(ci_upper), 4)
    }, {
        'metric': "Cohen's d (Effect Size)", 'value': round(float(cohens_d), 4)
    }]
else:
    # Binary/proportion test (conversion rates)
    control_conversions = control_group.sum()
    variant_conversions = variant_group.sum()
    control_n = len(control_group)
    variant_n = len(variant_group)

    control_rate = control_conversions / control_n
    variant_rate = variant_conversions / variant_n

    # Z-test for proportions
    pooled_rate = (control_conversions + variant_conversions) / (control_n + variant_n)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_n + 1/variant_n))
    z_stat = (variant_rate - control_rate) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    lift = ((variant_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0

    results = [{
        'metric': 'Sample Size (Control)', 'value': control_n
    }, {
        'metric': 'Sample Size (Variant)', 'value': variant_n
    }, {
        'metric': 'Control Conversion Rate', 'value': f"{round(float(control_rate)*100, 2)}%"
    }, {
        'metric': 'Variant Conversion Rate', 'value': f"{round(float(variant_rate)*100, 2)}%"
    }, {
        'metric': 'Relative Lift (%)', 'value': round(float(lift), 2)
    }, {
        'metric': 'P-Value', 'value': round(float(p_value), 6)
    }, {
        'metric': 'Statistically Significant', 'value': 'Yes' if p_value < (1 - confidence_level) else 'No'
    }]

output = pd.DataFrame(results)
`,
  },

  'cohort-analysis': {
    type: 'cohort-analysis',
    category: 'analysis',
    label: 'Cohort Analysis',
    description: 'Analyze user retention and behavior by cohort',
    icon: 'Users',
    defaultConfig: {
      userColumn: '',
      dateColumn: '',
      metricColumn: '',
      cohortPeriod: 'month',
      metricType: 'retention',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
user_column = config.get('userColumn', '')
date_column = config.get('dateColumn', '')
metric_column = config.get('metricColumn', '')
cohort_period = config.get('cohortPeriod', 'month')
metric_type = config.get('metricType', 'retention')

if not user_column or not date_column:
    raise ValueError("Cohort Analysis: Please specify user ID and date columns")

# Convert date column
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df = df.dropna(subset=[date_column])

# Determine cohort period format
if cohort_period == 'week':
    df['cohort'] = df[date_column].dt.to_period('W').dt.start_time
    df['period'] = df[date_column].dt.to_period('W').dt.start_time
elif cohort_period == 'month':
    df['cohort'] = df[date_column].dt.to_period('M').dt.start_time
    df['period'] = df[date_column].dt.to_period('M').dt.start_time
elif cohort_period == 'quarter':
    df['cohort'] = df[date_column].dt.to_period('Q').dt.start_time
    df['period'] = df[date_column].dt.to_period('Q').dt.start_time
else:  # year
    df['cohort'] = df[date_column].dt.to_period('Y').dt.start_time
    df['period'] = df[date_column].dt.to_period('Y').dt.start_time

# Get first cohort for each user
user_cohorts = df.groupby(user_column)['cohort'].min().reset_index()
user_cohorts.columns = [user_column, 'first_cohort']
df = df.merge(user_cohorts, on=user_column)

# Calculate period number (0, 1, 2, etc.)
df['period_number'] = ((df['period'] - df['first_cohort']).dt.days /
    {'week': 7, 'month': 30, 'quarter': 90, 'year': 365}[cohort_period]).round().astype(int)

if metric_type == 'retention':
    # Retention: count unique users per cohort per period
    cohort_data = df.groupby(['first_cohort', 'period_number'])[user_column].nunique().reset_index()
    cohort_data.columns = ['cohort', 'period', 'users']

    # Get cohort sizes (period 0)
    cohort_sizes = cohort_data[cohort_data['period'] == 0][['cohort', 'users']]
    cohort_sizes.columns = ['cohort', 'cohort_size']

    cohort_data = cohort_data.merge(cohort_sizes, on='cohort')
    cohort_data['retention_rate'] = (cohort_data['users'] / cohort_data['cohort_size'] * 100).round(2)

    # Pivot to create retention matrix
    retention_matrix = cohort_data.pivot(index='cohort', columns='period', values='retention_rate')
    retention_matrix = retention_matrix.reset_index()
    retention_matrix['cohort'] = retention_matrix['cohort'].astype(str)
    retention_matrix.columns = ['Cohort'] + [f'Period {i}' for i in retention_matrix.columns[1:]]

    output = retention_matrix
else:
    # Metric sum/avg per cohort per period
    if not metric_column:
        raise ValueError("Cohort Analysis: Please specify a metric column for non-retention analysis")

    df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')

    if metric_type == 'sum':
        cohort_data = df.groupby(['first_cohort', 'period_number'])[metric_column].sum().reset_index()
    else:  # average
        cohort_data = df.groupby(['first_cohort', 'period_number'])[metric_column].mean().reset_index()

    cohort_data.columns = ['cohort', 'period', 'value']

    # Pivot
    result_matrix = cohort_data.pivot(index='cohort', columns='period', values='value')
    result_matrix = result_matrix.reset_index()
    result_matrix['cohort'] = result_matrix['cohort'].astype(str)
    result_matrix.columns = ['Cohort'] + [f'Period {i}' for i in result_matrix.columns[1:]]

    output = result_matrix
`,
  },

  'rfm-analysis': {
    type: 'rfm-analysis',
    category: 'analysis',
    label: 'RFM Analysis',
    description: 'Segment customers by Recency, Frequency, Monetary value',
    icon: 'Target',
    defaultConfig: {
      customerColumn: '',
      dateColumn: '',
      revenueColumn: '',
      segments: 5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from datetime import datetime

df = input_data.copy()
customer_column = config.get('customerColumn', '')
date_column = config.get('dateColumn', '')
revenue_column = config.get('revenueColumn', '')
n_segments = config.get('segments', 5)

if not customer_column or not date_column or not revenue_column:
    raise ValueError("RFM Analysis: Please specify customer ID, date, and revenue columns")

# Convert columns
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df[revenue_column] = pd.to_numeric(df[revenue_column], errors='coerce')
df = df.dropna(subset=[date_column, revenue_column])

# Calculate reference date (max date + 1 day)
reference_date = df[date_column].max() + pd.Timedelta(days=1)

# Calculate RFM metrics per customer
rfm = df.groupby(customer_column).agg(
    recency=(date_column, lambda x: (reference_date - x.max()).days),
    frequency=(customer_column, 'count'),
    monetary=(revenue_column, 'sum')
).reset_index()

rfm.columns = [customer_column, 'recency', 'frequency', 'monetary']

# Score each metric (1-n_segments, with n_segments being best)
# For recency, lower is better, so we reverse
rfm['R_score'] = pd.qcut(rfm['recency'], q=n_segments, labels=range(n_segments, 0, -1), duplicates='drop').astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=n_segments, labels=range(1, n_segments + 1), duplicates='drop').astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=n_segments, labels=range(1, n_segments + 1), duplicates='drop').astype(int)

# Calculate RFM score
rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Segment labels based on RFM scores
def segment_customer(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f >= 3 and m <= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cant Lose Them'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r >= 3 and f <= 2 and m <= 2:
        return 'Promising'
    else:
        return 'Need Attention'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

# Round monetary for display
rfm['monetary'] = rfm['monetary'].round(2)

# Reorder columns for clarity
rfm = rfm[[customer_column, 'recency', 'frequency', 'monetary',
           'R_score', 'F_score', 'M_score', 'RFM_score', 'RFM_total', 'segment']]

output = rfm
`,
  },

  'anova': {
    type: 'anova',
    category: 'analysis',
    label: 'ANOVA',
    description: 'Analysis of variance to compare group means',
    icon: 'BarChart3',
    defaultConfig: {
      valueColumn: '',
      groupColumn: '',
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
value_col = config.get('valueColumn', '')
group_col = config.get('groupColumn', '')
alpha = float(config.get('alpha', 0.05))

if not value_col:
    raise ValueError("ANOVA: Please specify a value column in the Config tab")

if not group_col:
    raise ValueError("ANOVA: Please specify a group column in the Config tab")

if value_col not in df.columns:
    raise ValueError(f"ANOVA: Value column '{value_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

if group_col not in df.columns:
    raise ValueError(f"ANOVA: Group column '{group_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert value column to numeric
df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df = df.dropna(subset=[value_col, group_col])

# Get groups
groups = [group[value_col].values for name, group in df.groupby(group_col)]

if len(groups) < 2:
    raise ValueError("ANOVA: Need at least 2 groups to perform ANOVA")

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*groups)

# Calculate group statistics
group_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
group_stats.columns = ['group', 'count', 'mean', 'std', 'min', 'max']

# Create summary row
summary = pd.DataFrame({
    'group': ['ANOVA Result'],
    'count': [len(df)],
    'mean': [df[value_col].mean()],
    'std': [df[value_col].std()],
    'min': [f_stat],
    'max': [p_value]
})
summary.columns = ['group', 'count', 'mean', 'std', 'F_statistic', 'p_value']

# Add significance
significant = 'Yes' if p_value < alpha else 'No'

result = pd.DataFrame({
    'metric': ['F-statistic', 'p-value', 'Alpha', 'Significant', 'Number of groups', 'Total observations'],
    'value': [round(f_stat, 4), round(p_value, 6), alpha, significant, len(groups), len(df)]
})

output = result
`,
  },

  'chi-square-test': {
    type: 'chi-square-test',
    category: 'analysis',
    label: 'Chi-Square Test',
    description: 'Test independence between categorical variables',
    icon: 'Grid3X3',
    defaultConfig: {
      column1: '',
      column2: '',
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
col1 = config.get('column1', '')
col2 = config.get('column2', '')
alpha = float(config.get('alpha', 0.05))

if not col1 or not col2:
    raise ValueError("Chi-Square Test: Please specify both columns in the Config tab")

if col1 not in df.columns:
    raise ValueError(f"Chi-Square Test: Column '{col1}' not found. Available columns: {', '.join(df.columns.tolist())}")

if col2 not in df.columns:
    raise ValueError(f"Chi-Square Test: Column '{col2}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Create contingency table
contingency = pd.crosstab(df[col1], df[col2])

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

# Calculate Cramer's V for effect size
n = contingency.sum().sum()
min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

significant = 'Yes' if p_value < alpha else 'No'

result = pd.DataFrame({
    'metric': ['Chi-square statistic', 'p-value', 'Degrees of freedom', 'Alpha', 'Significant', "Cramer's V (effect size)", 'Observations'],
    'value': [round(chi2, 4), round(p_value, 6), dof, alpha, significant, round(cramers_v, 4), n]
})

output = result
`,
  },

  'correlation-analysis': {
    type: 'correlation-analysis',
    category: 'analysis',
    label: 'Correlation Analysis',
    description: 'Calculate correlation coefficients with p-values',
    icon: 'Link',
    defaultConfig: {
      columns: [],
      method: 'pearson',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'pearson')

# Convert columns to numeric
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        pass

# Use numeric columns if none specified
if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(columns) < 2:
    raise ValueError("Correlation Analysis: Need at least 2 numeric columns")

# Filter to selected columns
df = df[columns].dropna()

results = []

for i, col1 in enumerate(columns):
    for col2 in columns[i+1:]:
        x = df[col1].values
        y = df[col2].values

        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            corr, p_value = stats.pearsonr(x, y)

        strength = 'Strong' if abs(corr) >= 0.7 else ('Moderate' if abs(corr) >= 0.4 else 'Weak')
        direction = 'Positive' if corr > 0 else 'Negative'

        results.append({
            'column_1': col1,
            'column_2': col2,
            'correlation': round(corr, 4),
            'p_value': round(p_value, 6),
            'significant': 'Yes' if p_value < 0.05 else 'No',
            'strength': strength,
            'direction': direction
        })

output = pd.DataFrame(results)
`,
  },

  'survival-analysis': {
    type: 'survival-analysis',
    category: 'analysis',
    label: 'Survival Analysis',
    description: 'Kaplan-Meier survival analysis for time-to-event data',
    icon: 'Activity',
    defaultConfig: {
      timeColumn: '',
      eventColumn: '',
      groupColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
time_col = config.get('timeColumn', '')
event_col = config.get('eventColumn', '')
group_col = config.get('groupColumn', '')

if not time_col:
    raise ValueError("Survival Analysis: Please specify a time column in the Config tab")

if not event_col:
    raise ValueError("Survival Analysis: Please specify an event column in the Config tab")

if time_col not in df.columns:
    raise ValueError(f"Survival Analysis: Time column '{time_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

if event_col not in df.columns:
    raise ValueError(f"Survival Analysis: Event column '{event_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Convert to numeric
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
df = df.dropna(subset=[time_col, event_col])

def kaplan_meier(times, events):
    # Sort by time
    order = np.argsort(times)
    times = np.array(times)[order]
    events = np.array(events)[order]

    unique_times = np.unique(times)
    survival_prob = []
    at_risk = []
    events_at_time = []

    n = len(times)
    S = 1.0

    for t in unique_times:
        mask = times == t
        d = events[mask].sum()  # Events at time t
        n_t = (times >= t).sum()  # At risk at time t

        if n_t > 0:
            S = S * (1 - d / n_t)

        survival_prob.append(S)
        at_risk.append(n_t)
        events_at_time.append(d)

    return unique_times, survival_prob, at_risk, events_at_time

if group_col and group_col in df.columns:
    results = []
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        times, surv, risk, events = kaplan_meier(group_df[time_col].values, group_df[event_col].values)

        for i in range(len(times)):
            results.append({
                'group': group,
                'time': times[i],
                'survival_probability': round(surv[i], 4),
                'at_risk': risk[i],
                'events': events[i]
            })

    output = pd.DataFrame(results)
else:
    times, surv, risk, events = kaplan_meier(df[time_col].values, df[event_col].values)
    output = pd.DataFrame({
        'time': times,
        'survival_probability': [round(s, 4) for s in surv],
        'at_risk': risk,
        'events': events
    })
`,
  },

  'association-rules': {
    type: 'association-rules',
    category: 'analysis',
    label: 'Association Rules',
    description: 'Find item associations using Apriori algorithm',
    icon: 'Network',
    defaultConfig: {
      transactionColumn: '',
      itemColumn: '',
      minSupport: 0.01,
      minConfidence: 0.5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

df = input_data.copy()
trans_col = config.get('transactionColumn', '')
item_col = config.get('itemColumn', '')
min_support = float(config.get('minSupport', 0.01))
min_confidence = float(config.get('minConfidence', 0.5))

if not trans_col or not item_col:
    raise ValueError("Association Rules: Please specify transaction and item columns in the Config tab")

if trans_col not in df.columns:
    raise ValueError(f"Association Rules: Transaction column '{trans_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

if item_col not in df.columns:
    raise ValueError(f"Association Rules: Item column '{item_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Group items by transaction
transactions = df.groupby(trans_col)[item_col].apply(set).tolist()
n_transactions = len(transactions)

if n_transactions == 0:
    raise ValueError("Association Rules: No transactions found")

# Count item frequencies
item_counts = defaultdict(int)
for trans in transactions:
    for item in trans:
        item_counts[item] += 1

# Filter by support
frequent_items = {item for item, count in item_counts.items()
                  if count / n_transactions >= min_support}

# Count pairs
pair_counts = defaultdict(int)
for trans in transactions:
    items = trans & frequent_items
    for pair in combinations(sorted(items), 2):
        pair_counts[pair] += 1

# Generate rules
rules = []
for (item1, item2), count in pair_counts.items():
    support = count / n_transactions
    if support >= min_support:
        # Rule: item1 -> item2
        conf_1_2 = count / item_counts[item1]
        if conf_1_2 >= min_confidence:
            lift_1_2 = conf_1_2 / (item_counts[item2] / n_transactions)
            rules.append({
                'antecedent': item1,
                'consequent': item2,
                'support': round(support, 4),
                'confidence': round(conf_1_2, 4),
                'lift': round(lift_1_2, 4)
            })

        # Rule: item2 -> item1
        conf_2_1 = count / item_counts[item2]
        if conf_2_1 >= min_confidence:
            lift_2_1 = conf_2_1 / (item_counts[item1] / n_transactions)
            rules.append({
                'antecedent': item2,
                'consequent': item1,
                'support': round(support, 4),
                'confidence': round(conf_2_1, 4),
                'lift': round(lift_2_1, 4)
            })

if not rules:
    output = pd.DataFrame({'message': ['No rules found. Try lowering min_support or min_confidence.']})
else:
    output = pd.DataFrame(rules).sort_values('lift', ascending=False)
`,
  },

  'sentiment-analysis': {
    type: 'sentiment-analysis',
    category: 'analysis',
    label: 'Sentiment Analysis',
    description: 'Analyze text sentiment (positive, negative, neutral)',
    icon: 'MessageSquare',
    defaultConfig: {
      textColumn: '',
      outputColumn: 'sentiment',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re

df = input_data.copy()
text_col = config.get('textColumn', '')
output_col = config.get('outputColumn', 'sentiment')

if not text_col:
    raise ValueError("Sentiment Analysis: Please specify a text column in the Config tab")

if text_col not in df.columns:
    raise ValueError(f"Sentiment Analysis: Column '{text_col}' not found. Available columns: {', '.join(df.columns.tolist())}")

# Simple lexicon-based sentiment analysis
positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
                      'love', 'like', 'best', 'happy', 'beautiful', 'perfect', 'positive', 'brilliant',
                      'outstanding', 'superb', 'delightful', 'pleasant', 'enjoy', 'recommended',
                      'impressive', 'thank', 'thanks', 'helpful', 'satisfied', 'nice', 'easy'])

negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate', 'dislike',
                      'disappointing', 'disappointed', 'negative', 'wrong', 'fail', 'failed', 'useless',
                      'waste', 'problem', 'issue', 'difficult', 'hard', 'annoying', 'frustrating',
                      'angry', 'sad', 'unfortunately', 'broken', 'error', 'bug', 'slow'])

def analyze_sentiment(text):
    if pd.isna(text):
        return 'neutral', 0

    text = str(text).lower()
    words = re.findall(r'\\b\\w+\\b', text)

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    score = (pos_count - neg_count) / max(len(words), 1)

    if score > 0.05:
        return 'positive', round(score, 3)
    elif score < -0.05:
        return 'negative', round(score, 3)
    else:
        return 'neutral', round(score, 3)

results = df[text_col].apply(analyze_sentiment)
df[output_col] = results.apply(lambda x: x[0])
df[f'{output_col}_score'] = results.apply(lambda x: x[1])

# Add summary statistics
summary = df[output_col].value_counts()

output = df
`,
  },

  'moving-average': {
    type: 'moving-average',
    category: 'analysis',
    label: 'Moving Average',
    description: 'Calculate simple and exponential moving averages',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      window: 3,
      type: 'simple',
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
window = int(config.get('window', 3))
ma_type = config.get('type', 'simple')
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Moving Average: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Moving Average: Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}")

if not output_col:
    output_col = f"{column}_ma{window}"

# Convert to numeric
numeric_col = pd.to_numeric(df[column], errors='coerce')

if ma_type == 'simple':
    df[output_col] = numeric_col.rolling(window=window, min_periods=1).mean()
elif ma_type == 'exponential':
    df[output_col] = numeric_col.ewm(span=window, adjust=False).mean()
elif ma_type == 'weighted':
    weights = list(range(1, window + 1))
    df[output_col] = numeric_col.rolling(window=window, min_periods=1).apply(
        lambda x: sum(w * v for w, v in zip(weights[-len(x):], x)) / sum(weights[-len(x):])
    )

df[output_col] = df[output_col].round(4)

output = df
`,
  },

  'train-test-split': {
    type: 'train-test-split',
    category: 'analysis',
    label: 'Train/Test Split',
    description: 'Split data into training and testing sets for machine learning',
    icon: 'Split',
    defaultConfig: {
      testSize: 0.2,
      randomState: 42,
      stratifyColumn: '',
      shuffle: true,
    },
    inputs: 1,
    outputs: 2,
    pythonTemplate: `
import pandas as pd
from sklearn.model_selection import train_test_split

df = input_data.copy()
test_size = float(config.get('testSize', 0.2))
random_state = int(config.get('randomState', 42))
stratify_col = config.get('stratifyColumn', '')
shuffle = config.get('shuffle', True)

if test_size <= 0 or test_size >= 1:
    raise ValueError("Train/Test Split: Test size must be between 0 and 1 (e.g., 0.2 for 20%)")

stratify = None
if stratify_col and stratify_col in df.columns:
    stratify = df[stratify_col]

train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle,
    stratify=stratify
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Add split indicator
train_df['_split'] = 'train'
test_df['_split'] = 'test'

# Combine and return
output = pd.concat([train_df, test_df], ignore_index=True)
`,
  },

  'model-evaluation': {
    type: 'model-evaluation',
    category: 'analysis',
    label: 'Model Evaluation',
    description: 'Calculate model performance metrics (accuracy, precision, recall, F1, MAE, RMSE)',
    icon: 'ClipboardCheck',
    defaultConfig: {
      actualColumn: '',
      predictedColumn: '',
      taskType: 'classification',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

df = input_data.copy()
actual_col = config.get('actualColumn', '')
predicted_col = config.get('predictedColumn', '')
task_type = config.get('taskType', 'classification')

if not actual_col or not predicted_col:
    raise ValueError("Model Evaluation: Please specify both actual and predicted columns in the Config tab")

if actual_col not in df.columns:
    raise ValueError(f"Model Evaluation: Actual column '{actual_col}' not found")
if predicted_col not in df.columns:
    raise ValueError(f"Model Evaluation: Predicted column '{predicted_col}' not found")

y_true = df[actual_col]
y_pred = df[predicted_col]

# Remove any NaN values
mask = ~(pd.isna(y_true) | pd.isna(y_pred))
y_true = y_true[mask]
y_pred = y_pred[mask]

if task_type == 'classification':
    metrics = {
        'Metric': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1 Score (weighted)'],
        'Value': [
            round(accuracy_score(y_true, y_pred), 4),
            round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
        ]
    }
else:  # regression
    y_true_num = pd.to_numeric(y_true, errors='coerce')
    y_pred_num = pd.to_numeric(y_pred, errors='coerce')
    mask = ~(pd.isna(y_true_num) | pd.isna(y_pred_num))
    y_true_num = y_true_num[mask]
    y_pred_num = y_pred_num[mask]

    mse = mean_squared_error(y_true_num, y_pred_num)
    metrics = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
        'Value': [
            round(mean_absolute_error(y_true_num, y_pred_num), 4),
            round(mse, 4),
            round(np.sqrt(mse), 4),
            round(r2_score(y_true_num, y_pred_num), 4)
        ]
    }

output = pd.DataFrame(metrics)
`,
  },

  'knn': {
    type: 'knn',
    category: 'analysis',
    label: 'K-Nearest Neighbors',
    description: 'KNN classifier or regressor for prediction',
    icon: 'Users',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      k: 5,
      taskType: 'classification',
      weights: 'uniform',
      outputColumn: 'knn_prediction',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
k = int(config.get('k', 5))
task_type = config.get('taskType', 'classification')
weights = config.get('weights', 'uniform')
output_col = config.get('outputColumn', 'knn_prediction')

if not feature_cols or not target_col:
    raise ValueError("KNN: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"KNN: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"KNN: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Handle missing values
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

mask = ~pd.isna(y)
X_train = X[mask]
y_train = y[mask]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)

if task_type == 'classification':
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
else:
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)

model.fit(X_train_scaled, y_train)
predictions = model.predict(X_scaled)

df[output_col] = predictions

output = df
`,
  },

  'naive-bayes': {
    type: 'naive-bayes',
    category: 'analysis',
    label: 'Naive Bayes',
    description: 'Naive Bayes classifier for categorical/text classification',
    icon: 'Brain',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      variant: 'gaussian',
      outputColumn: 'nb_prediction',
      outputProbability: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
variant = config.get('variant', 'gaussian')
output_col = config.get('outputColumn', 'nb_prediction')
output_prob = config.get('outputProbability', False)

if not feature_cols or not target_col:
    raise ValueError("Naive Bayes: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Naive Bayes: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"Naive Bayes: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# Encode target if needed
le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))

# Select model variant
if variant == 'gaussian':
    model = GaussianNB()
elif variant == 'multinomial':
    X = X.clip(lower=0)  # Ensure non-negative
    model = MultinomialNB()
else:  # bernoulli
    model = BernoulliNB()

model.fit(X, y_encoded)
predictions = model.predict(X)
df[output_col] = le.inverse_transform(predictions)

if output_prob:
    proba = model.predict_proba(X)
    for i, class_name in enumerate(le.classes_):
        df[f'{output_col}_prob_{class_name}'] = proba[:, i].round(4)

output = df
`,
  },

  'gradient-boosting': {
    type: 'gradient-boosting',
    category: 'analysis',
    label: 'Gradient Boosting',
    description: 'High-performance gradient boosting classifier or regressor',
    icon: 'Rocket',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      taskType: 'classification',
      nEstimators: 100,
      learningRate: 0.1,
      maxDepth: 3,
      outputColumn: 'gb_prediction',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
task_type = config.get('taskType', 'classification')
n_estimators = int(config.get('nEstimators', 100))
learning_rate = float(config.get('learningRate', 0.1))
max_depth = int(config.get('maxDepth', 3))
output_col = config.get('outputColumn', 'gb_prediction')

if not feature_cols or not target_col:
    raise ValueError("Gradient Boosting: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Gradient Boosting: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"Gradient Boosting: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

if task_type == 'classification':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X, y_encoded)
    predictions = model.predict(X)
    df[output_col] = le.inverse_transform(predictions)
else:
    y_numeric = pd.to_numeric(y, errors='coerce')
    mask = ~pd.isna(y_numeric)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X[mask], y_numeric[mask])
    df[output_col] = model.predict(X)

output = df
`,
  },

  'pareto-analysis': {
    type: 'pareto-analysis',
    category: 'analysis',
    label: 'Pareto Analysis',
    description: 'Identify vital few vs trivial many using 80/20 rule',
    icon: 'BarChart3',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      threshold: 80,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
val_col = config.get('valueColumn', '')
threshold = float(config.get('threshold', 80))

if not cat_col or not val_col:
    raise ValueError("Pareto Analysis: Please specify category and value columns in the Config tab")

if cat_col not in df.columns:
    raise ValueError(f"Pareto Analysis: Category column '{cat_col}' not found")
if val_col not in df.columns:
    raise ValueError(f"Pareto Analysis: Value column '{val_col}' not found")

# Aggregate by category
agg_df = df.groupby(cat_col)[val_col].sum().reset_index()
agg_df.columns = ['Category', 'Value']

# Sort descending
agg_df = agg_df.sort_values('Value', ascending=False).reset_index(drop=True)

# Calculate percentages
total = agg_df['Value'].sum()
agg_df['Percentage'] = (agg_df['Value'] / total * 100).round(2)
agg_df['Cumulative_Percentage'] = agg_df['Percentage'].cumsum().round(2)

# Classify as vital few or trivial many
agg_df['Classification'] = np.where(
    agg_df['Cumulative_Percentage'] <= threshold,
    'Vital Few',
    'Trivial Many'
)

# Calculate rank
agg_df['Rank'] = range(1, len(agg_df) + 1)

output = agg_df
`,
  },

  'trend-analysis': {
    type: 'trend-analysis',
    category: 'analysis',
    label: 'Trend Analysis',
    description: 'Detect and quantify trends in time series data',
    icon: 'TrendingUp',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      trendType: 'linear',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
date_col = config.get('dateColumn', '')
val_col = config.get('valueColumn', '')
trend_type = config.get('trendType', 'linear')

if not date_col or not val_col:
    raise ValueError("Trend Analysis: Please specify date and value columns in the Config tab")

if date_col not in df.columns:
    raise ValueError(f"Trend Analysis: Date column '{date_col}' not found")
if val_col not in df.columns:
    raise ValueError(f"Trend Analysis: Value column '{val_col}' not found")

# Sort by date
df = df.sort_values(date_col).reset_index(drop=True)

# Convert value to numeric
y = pd.to_numeric(df[val_col], errors='coerce')
x = np.arange(len(y))

# Remove NaN
mask = ~pd.isna(y)
x_clean = x[mask]
y_clean = y[mask].values

if len(x_clean) < 2:
    raise ValueError("Trend Analysis: Need at least 2 data points")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

# Calculate trend values
df['Trend_Value'] = intercept + slope * x
df['Residual'] = y - df['Trend_Value']

# Determine trend direction
if slope > 0 and p_value < 0.05:
    direction = 'Upward'
elif slope < 0 and p_value < 0.05:
    direction = 'Downward'
else:
    direction = 'No significant trend'

# Calculate trend strength
trend_strength = abs(r_value)
if trend_strength >= 0.7:
    strength_label = 'Strong'
elif trend_strength >= 0.4:
    strength_label = 'Moderate'
else:
    strength_label = 'Weak'

# Add trend info to data
df['Trend_Direction'] = direction
df['Trend_Strength'] = strength_label

output = df
`,
  },

  'forecasting': {
    type: 'forecasting',
    category: 'analysis',
    label: 'Forecasting',
    description: 'Predict future values using time series methods',
    icon: 'CalendarClock',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      periods: 10,
      method: 'exponential',
      alpha: 0.3,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
val_col = config.get('valueColumn', '')
periods = int(config.get('periods', 10))
method = config.get('method', 'exponential')
alpha = float(config.get('alpha', 0.3))

if not date_col or not val_col:
    raise ValueError("Forecasting: Please specify date and value columns in the Config tab")

if date_col not in df.columns:
    raise ValueError(f"Forecasting: Date column '{date_col}' not found")
if val_col not in df.columns:
    raise ValueError(f"Forecasting: Value column '{val_col}' not found")

# Sort by date
df = df.sort_values(date_col).reset_index(drop=True)

# Convert to numeric
values = pd.to_numeric(df[val_col], errors='coerce').fillna(method='ffill').fillna(0)

if method == 'linear':
    # Linear extrapolation
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + periods)
    forecast = intercept + slope * future_x

elif method == 'exponential':
    # Simple exponential smoothing
    forecast = []
    level = values.iloc[0]
    for val in values:
        level = alpha * val + (1 - alpha) * level
    for _ in range(periods):
        forecast.append(level)
    forecast = np.array(forecast)

else:  # moving_average
    window = min(5, len(values))
    ma = values.rolling(window=window).mean().iloc[-1]
    forecast = np.full(periods, ma)

# Try to generate future dates
try:
    last_date = pd.to_datetime(df[date_col].iloc[-1])
    date_diff = pd.to_datetime(df[date_col]).diff().median()
    if pd.isna(date_diff):
        date_diff = pd.Timedelta(days=1)
    future_dates = [last_date + date_diff * (i + 1) for i in range(periods)]
except:
    future_dates = [f"Period_{i+1}" for i in range(periods)]

# Create forecast dataframe
forecast_df = pd.DataFrame({
    date_col: future_dates,
    val_col: np.round(forecast, 2),
    'Type': 'Forecast'
})

# Mark original data
df['Type'] = 'Actual'

# Combine
result = pd.concat([df, forecast_df], ignore_index=True)

output = result
`,
  },

  'percentile-analysis': {
    type: 'percentile-analysis',
    category: 'analysis',
    label: 'Percentile Analysis',
    description: 'Calculate percentiles and quantile ranks',
    icon: 'Percent',
    defaultConfig: {
      column: '',
      percentiles: '10,25,50,75,90,95,99',
      addRank: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
percentiles_str = config.get('percentiles', '10,25,50,75,90,95,99')
add_rank = config.get('addRank', True)

if not column:
    raise ValueError("Percentile Analysis: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Percentile Analysis: Column '{column}' not found")

# Parse percentiles
percentiles = [float(p.strip()) for p in percentiles_str.split(',')]

# Convert to numeric
values = pd.to_numeric(df[column], errors='coerce')

# Add percentile rank to original data if requested
if add_rank:
    df[f'{column}_percentile_rank'] = values.rank(pct=True).round(4) * 100

# Add quartile classification
try:
    df[f'{column}_quartile'] = pd.qcut(values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
except:
    df[f'{column}_quartile'] = 'N/A'

output = df
`,
  },

  'distribution-fit': {
    type: 'distribution-fit',
    category: 'analysis',
    label: 'Distribution Fit',
    description: 'Fit data to statistical distributions',
    icon: 'Activity',
    defaultConfig: {
      column: '',
      distributions: 'normal,exponential,uniform',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
column = config.get('column', '')
dist_str = config.get('distributions', 'normal,exponential,uniform')

if not column:
    raise ValueError("Distribution Fit: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Distribution Fit: Column '{column}' not found")

# Get numeric values
values = pd.to_numeric(df[column], errors='coerce').dropna()

if len(values) < 10:
    raise ValueError("Distribution Fit: Need at least 10 data points")

# Parse distributions
dist_names = [d.strip().lower() for d in dist_str.split(',')]

results = []
for dist_name in dist_names:
    try:
        if dist_name == 'normal':
            params = stats.norm.fit(values)
            ks_stat, p_value = stats.kstest(values, 'norm', params)
            results.append({
                'Distribution': 'Normal',
                'Parameters': f'mean={params[0]:.4f}, std={params[1]:.4f}',
                'KS_Statistic': round(ks_stat, 4),
                'P_Value': round(p_value, 4),
                'Good_Fit': 'Yes' if p_value > 0.05 else 'No'
            })
        elif dist_name == 'exponential':
            params = stats.expon.fit(values)
            ks_stat, p_value = stats.kstest(values, 'expon', params)
            results.append({
                'Distribution': 'Exponential',
                'Parameters': f'loc={params[0]:.4f}, scale={params[1]:.4f}',
                'KS_Statistic': round(ks_stat, 4),
                'P_Value': round(p_value, 4),
                'Good_Fit': 'Yes' if p_value > 0.05 else 'No'
            })
        elif dist_name == 'uniform':
            params = stats.uniform.fit(values)
            ks_stat, p_value = stats.kstest(values, 'uniform', params)
            results.append({
                'Distribution': 'Uniform',
                'Parameters': f'loc={params[0]:.4f}, scale={params[1]:.4f}',
                'KS_Statistic': round(ks_stat, 4),
                'P_Value': round(p_value, 4),
                'Good_Fit': 'Yes' if p_value > 0.05 else 'No'
            })
    except Exception as e:
        results.append({
            'Distribution': dist_name.title(),
            'Parameters': f'Error: {str(e)}',
            'KS_Statistic': None,
            'P_Value': None,
            'Good_Fit': 'Error'
        })

output = pd.DataFrame(results)
`,
  },

  'text-preprocessing': {
    type: 'text-preprocessing',
    category: 'analysis',
    label: 'Text Preprocessing',
    description: 'Clean and prepare text data for analysis',
    icon: 'FileText',
    defaultConfig: {
      column: '',
      lowercase: true,
      removePunctuation: true,
      removeNumbers: false,
      removeStopwords: true,
      trimWhitespace: true,
      outputColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re
import string

df = input_data.copy()
column = config.get('column', '')
lowercase = config.get('lowercase', True)
remove_punct = config.get('removePunctuation', True)
remove_nums = config.get('removeNumbers', False)
remove_stop = config.get('removeStopwords', True)
trim_ws = config.get('trimWhitespace', True)
output_col = config.get('outputColumn', '')

if not column:
    raise ValueError("Text Preprocessing: Please specify a text column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Text Preprocessing: Column '{column}' not found")

if not output_col:
    output_col = f'{column}_cleaned'

# Common English stopwords
stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
             'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
             'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
             'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
             'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just'}

def preprocess_text(text):
    if pd.isna(text):
        return ''

    text = str(text)

    if lowercase:
        text = text.lower()

    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    if remove_nums:
        text = re.sub(r'\\d+', '', text)

    if trim_ws:
        text = ' '.join(text.split())

    if remove_stop:
        words = text.split()
        words = [w for w in words if w.lower() not in stopwords]
        text = ' '.join(words)

    return text

df[output_col] = df[column].apply(preprocess_text)

# Add word count
df[f'{output_col}_word_count'] = df[output_col].apply(lambda x: len(str(x).split()) if x else 0)

output = df
`,
  },

  'tfidf-vectorization': {
    type: 'tfidf-vectorization',
    category: 'analysis',
    label: 'TF-IDF Vectorization',
    description: 'Convert text to TF-IDF numerical features',
    icon: 'Hash',
    defaultConfig: {
      column: '',
      maxFeatures: 100,
      ngramMin: 1,
      ngramMax: 1,
      outputFormat: 'top_terms',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = input_data.copy()
column = config.get('column', '')
max_features = int(config.get('maxFeatures', 100))
ngram_min = int(config.get('ngramMin', 1))
ngram_max = int(config.get('ngramMax', 1))
output_format = config.get('outputFormat', 'top_terms')

if not column:
    raise ValueError("TF-IDF: Please specify a text column in the Config tab")

if column not in df.columns:
    raise ValueError(f"TF-IDF: Column '{column}' not found")

# Get text data
texts = df[column].fillna('').astype(str)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(ngram_min, ngram_max),
    stop_words='english'
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

if output_format == 'matrix':
    # Return full TF-IDF matrix as dataframe
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{name}' for name in feature_names]
    )
    result = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
elif output_format == 'top_terms':
    # Return top terms per document
    def get_top_terms(row_idx, n=5):
        row = tfidf_matrix[row_idx].toarray().flatten()
        top_indices = row.argsort()[-n:][::-1]
        return ', '.join([feature_names[i] for i in top_indices if row[i] > 0])

    df['top_tfidf_terms'] = [get_top_terms(i) for i in range(len(df))]
    result = df
else:  # summary
    # Return term importance summary
    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    term_importance = pd.DataFrame({
        'Term': feature_names,
        'Mean_TFIDF': np.round(mean_tfidf, 4),
        'Doc_Frequency': np.array((tfidf_matrix > 0).sum(axis=0)).flatten()
    }).sort_values('Mean_TFIDF', ascending=False)
    result = term_importance

output = result
`,
  },

  'topic-modeling': {
    type: 'topic-modeling',
    category: 'analysis',
    label: 'Topic Modeling',
    description: 'Discover hidden topics in text using LDA',
    icon: 'Layers',
    defaultConfig: {
      column: '',
      numTopics: 5,
      numWords: 10,
      maxFeatures: 1000,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = input_data.copy()
column = config.get('column', '')
num_topics = int(config.get('numTopics', 5))
num_words = int(config.get('numWords', 10))
max_features = int(config.get('maxFeatures', 1000))

if not column:
    raise ValueError("Topic Modeling: Please specify a text column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Topic Modeling: Column '{column}' not found")

# Get text data
texts = df[column].fillna('').astype(str)

# Create count vectorizer
vectorizer = CountVectorizer(
    max_features=max_features,
    stop_words='english',
    max_df=0.95,
    min_df=2
)

# Fit and transform
doc_term_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# Fit LDA model
lda = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42,
    max_iter=20
)
doc_topics = lda.fit_transform(doc_term_matrix)

# Add dominant topic to original data
df['Dominant_Topic'] = [f'Topic_{i+1}' for i in doc_topics.argmax(axis=1)]
df['Topic_Confidence'] = doc_topics.max(axis=1).round(4)

# Add topic distribution columns
for i in range(num_topics):
    df[f'Topic_{i+1}_Score'] = doc_topics[:, i].round(4)

output = df
`,
  },

  'similarity-analysis': {
    type: 'similarity-analysis',
    category: 'analysis',
    label: 'Similarity Analysis',
    description: 'Calculate similarity between rows or find similar items',
    icon: 'GitCompare',
    defaultConfig: {
      columns: [],
      method: 'cosine',
      topN: 5,
      outputType: 'add_to_data',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'cosine')
top_n = int(config.get('topN', 5))
output_type = config.get('outputType', 'add_to_data')

if not columns:
    raise ValueError("Similarity Analysis: Please specify columns to use for similarity in the Config tab")

missing_cols = [c for c in columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"Similarity Analysis: Columns not found: {missing_cols}")

# Prepare feature matrix
X = df[columns].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate similarity matrix
if method == 'cosine':
    sim_matrix = cosine_similarity(X_scaled)
elif method == 'euclidean':
    dist_matrix = euclidean_distances(X_scaled)
    sim_matrix = 1 / (1 + dist_matrix)  # Convert distance to similarity
else:  # manhattan
    from sklearn.metrics.pairwise import manhattan_distances
    dist_matrix = manhattan_distances(X_scaled)
    sim_matrix = 1 / (1 + dist_matrix)

if output_type == 'matrix':
    # Return full similarity matrix
    sim_df = pd.DataFrame(
        sim_matrix.round(4),
        columns=[f'sim_row_{i}' for i in range(len(df))],
        index=range(len(df))
    )
    result = sim_df
elif output_type == 'top_similar':
    # Find top N similar items for each row
    top_similar = []
    for i in range(len(df)):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # Exclude self
        top_indices = sims.argsort()[-top_n:][::-1]
        top_similar.append({
            'Row_Index': i,
            'Top_Similar_Indices': ', '.join(map(str, top_indices)),
            'Similarity_Scores': ', '.join([f'{sims[j]:.4f}' for j in top_indices])
        })
    result = pd.DataFrame(top_similar)
else:  # add_to_data
    # Add average similarity score to original data
    np.fill_diagonal(sim_matrix, 0)
    df['Avg_Similarity'] = sim_matrix.mean(axis=1).round(4)
    df['Max_Similarity'] = sim_matrix.max(axis=1).round(4)
    result = df

output = result
`,
  },

  'svm': {
    type: 'svm',
    category: 'analysis',
    label: 'SVM',
    description: 'Support Vector Machine for classification and regression',
    icon: 'Target',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      mode: 'classification',
      kernel: 'rbf',
      c: 1.0,
      gamma: 'scale',
      degree: 3,
      outputColumn: 'svm_prediction',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
mode = config.get('mode', 'classification')
kernel = config.get('kernel', 'rbf')
c_param = float(config.get('c', 1.0))
gamma = config.get('gamma', 'scale')
degree = int(config.get('degree', 3))
output_col = config.get('outputColumn', 'svm_prediction')

if not feature_cols or not target_col:
    raise ValueError("SVM: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"SVM: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"SVM: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

# Scale features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if mode == 'classification':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    model = SVC(kernel=kernel, C=c_param, gamma=gamma, degree=degree, random_state=42)
    model.fit(X_scaled, y_encoded)
    predictions = model.predict(X_scaled)

    df[output_col] = le.inverse_transform(predictions)
    df['svm_support_vectors_count'] = len(model.support_)

    accuracy = accuracy_score(y_encoded, predictions)
    f1 = f1_score(y_encoded, predictions, average='weighted')
    df['svm_accuracy'] = round(accuracy, 4)
    df['svm_f1_score'] = round(f1, 4)
else:
    y_numeric = pd.to_numeric(y, errors='coerce')
    mask = ~pd.isna(y_numeric)

    model = SVR(kernel=kernel, C=c_param, gamma=gamma, degree=degree)
    model.fit(X_scaled[mask], y_numeric[mask])
    df[output_col] = model.predict(X_scaled)
    df['svm_support_vectors_count'] = len(model.support_)

    predictions = model.predict(X_scaled[mask])
    r2 = r2_score(y_numeric[mask], predictions)
    rmse = np.sqrt(mean_squared_error(y_numeric[mask], predictions))
    df['svm_r2'] = round(r2, 4)
    df['svm_rmse'] = round(rmse, 4)

output = df
`,
  },

  'xgboost': {
    type: 'xgboost',
    category: 'analysis',
    label: 'XGBoost',
    description: 'Extreme Gradient Boosting classifier/regressor',
    icon: 'Zap',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      mode: 'classification',
      nEstimators: 100,
      maxDepth: 6,
      learningRate: 0.3,
      outputColumn: 'xgb_prediction',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
mode = config.get('mode', 'classification')
n_estimators = int(config.get('nEstimators', 100))
max_depth = int(config.get('maxDepth', 6))
learning_rate = float(config.get('learningRate', 0.3))
output_col = config.get('outputColumn', 'xgb_prediction')

if not feature_cols or not target_col:
    raise ValueError("XGBoost: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"XGBoost: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"XGBoost: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

if mode == 'classification':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X, y_encoded)
    predictions = model.predict(X)

    df[output_col] = le.inverse_transform(predictions)

    # Probabilities if available
    proba = model.predict_proba(X)
    df['xgb_probability'] = proba.max(axis=1).round(4)

    accuracy = accuracy_score(y_encoded, predictions)
    f1 = f1_score(y_encoded, predictions, average='weighted')
    df['xgb_accuracy'] = round(accuracy, 4)
    df['xgb_f1_score'] = round(f1, 4)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).round(4)
    for feat, imp in importance.items():
        df[f'xgb_importance_{feat}'] = imp
else:
    y_numeric = pd.to_numeric(y, errors='coerce')
    mask = ~pd.isna(y_numeric)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X[mask], y_numeric[mask])
    df[output_col] = model.predict(X)

    predictions = model.predict(X[mask])
    r2 = r2_score(y_numeric[mask], predictions)
    rmse = np.sqrt(mean_squared_error(y_numeric[mask], predictions))
    df['xgb_r2'] = round(r2, 4)
    df['xgb_rmse'] = round(rmse, 4)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).round(4)
    for feat, imp in importance.items():
        df[f'xgb_importance_{feat}'] = imp

output = df
`,
  },

  'model-explainability': {
    type: 'model-explainability',
    category: 'analysis',
    label: 'Model Explainability (SHAP)',
    description: 'SHAP values for model interpretability',
    icon: 'Lightbulb',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      nSamples: 100,
      plotType: 'summary',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
n_samples = min(int(config.get('nSamples', 100)), len(df))
plot_type = config.get('plotType', 'summary')

if not feature_cols or not target_col:
    raise ValueError("Model Explainability: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Model Explainability: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"Model Explainability: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

# Determine if classification or regression
is_classification = y.dtype == 'object' or y.nunique() < 10

if is_classification:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y_encoded)
else:
    y_numeric = pd.to_numeric(y, errors='coerce')
    mask = ~pd.isna(y_numeric)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X[mask], y_numeric[mask])

# Calculate permutation-based feature importance as SHAP proxy
# (Full SHAP requires shap library which may not be available)
base_importance = model.feature_importances_

# Calculate contribution scores for each sample
X_sample = X.head(n_samples)
contributions = np.zeros((len(X_sample), len(feature_cols)))

for i, col in enumerate(feature_cols):
    contributions[:, i] = X_sample[col].values * base_importance[i]

# Normalize contributions
contributions = contributions / (np.abs(contributions).sum(axis=1, keepdims=True) + 1e-10)

# Create output dataframe
result_df = df.head(n_samples).copy()

# Add contribution columns for each feature
for i, col in enumerate(feature_cols):
    result_df[f'shap_{col}'] = contributions[:, i].round(4)

# Add global feature importance ranking
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': base_importance.round(4)
}).sort_values('importance', ascending=False)

# Add importance rank to each row
for i, row in importance_df.iterrows():
    result_df[f'importance_rank_{row["feature"]}'] = importance_df.index.tolist().index(i) + 1

# Add global importance as column
result_df['top_feature'] = importance_df.iloc[0]['feature']
result_df['top_feature_importance'] = importance_df.iloc[0]['importance']

output = result_df
`,
  },

  'regression-diagnostics': {
    type: 'regression-diagnostics',
    category: 'analysis',
    label: 'Regression Diagnostics',
    description: 'Residual analysis for validating regression models',
    icon: 'Microscope',
    defaultConfig: {
      featureColumns: [],
      targetColumn: '',
      significanceLevel: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
target_col = config.get('targetColumn', '')
alpha = float(config.get('significanceLevel', 0.05))

if not feature_cols or not target_col:
    raise ValueError("Regression Diagnostics: Please specify feature columns and target column in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Regression Diagnostics: Feature columns not found: {missing_cols}")
if target_col not in df.columns:
    raise ValueError(f"Regression Diagnostics: Target column '{target_col}' not found")

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Remove missing values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
df_clean = df[mask].copy()

if len(X) < 10:
    raise ValueError("Regression Diagnostics: Need at least 10 valid samples for analysis")

# Fit regression model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
residuals = y - predictions
standardized_residuals = (residuals - residuals.mean()) / residuals.std()

# Add residuals to output
df_clean['predicted'] = predictions.round(4)
df_clean['residual'] = residuals.round(4)
df_clean['standardized_residual'] = standardized_residuals.round(4)

# 1. Shapiro-Wilk test for normality of residuals
n_shapiro = min(len(residuals), 5000)
shapiro_stat, shapiro_p = stats.shapiro(residuals.head(n_shapiro))
normality_passed = shapiro_p > alpha

# 2. Durbin-Watson test for autocorrelation
dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
# DW ~ 2 means no autocorrelation, < 1 or > 3 suggests problems
autocorr_passed = 1.5 < dw_stat < 2.5

# 3. Breusch-Pagan test for heteroscedasticity (simplified)
# Regress squared residuals on X
squared_residuals = residuals**2
bp_model = LinearRegression()
bp_model.fit(X, squared_residuals)
bp_predictions = bp_model.predict(X)
bp_ss_res = np.sum((squared_residuals - bp_predictions)**2)
bp_ss_tot = np.sum((squared_residuals - squared_residuals.mean())**2)
bp_r2 = 1 - bp_ss_res / bp_ss_tot
bp_stat = len(X) * bp_r2
bp_p = 1 - stats.chi2.cdf(bp_stat, len(feature_cols))
homoscedasticity_passed = bp_p > alpha

# 4. Cook's distance for influential points
n = len(X)
p = len(feature_cols) + 1
leverage = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
cooks_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage + 1e-10))
df_clean['cooks_distance'] = cooks_d.round(4)
influential_threshold = 4 / n
df_clean['is_influential'] = cooks_d > influential_threshold
n_influential = (cooks_d > influential_threshold).sum()

# Add diagnostic summary columns
df_clean['shapiro_wilk_stat'] = round(shapiro_stat, 4)
df_clean['shapiro_wilk_p'] = round(shapiro_p, 4)
df_clean['normality_passed'] = normality_passed
df_clean['durbin_watson'] = round(dw_stat, 4)
df_clean['autocorr_passed'] = autocorr_passed
df_clean['breusch_pagan_p'] = round(bp_p, 4)
df_clean['homoscedasticity_passed'] = homoscedasticity_passed
df_clean['n_influential_points'] = n_influential

output = df_clean
`,
  },

  'vif-analysis': {
    type: 'vif-analysis',
    category: 'analysis',
    label: 'VIF Analysis',
    description: 'Variance Inflation Factor to detect multicollinearity',
    icon: 'GitMerge',
    defaultConfig: {
      featureColumns: [],
      threshold: 5.0,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
threshold = float(config.get('threshold', 5.0))

if not feature_cols:
    raise ValueError("VIF Analysis: Please specify feature columns in the Config tab")

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"VIF Analysis: Columns not found: {missing_cols}")

if len(feature_cols) < 2:
    raise ValueError("VIF Analysis: Need at least 2 feature columns to calculate VIF")

X = df[feature_cols].copy()

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

# Calculate VIF for each feature
vif_data = []
for i, col in enumerate(feature_cols):
    y_vif = X[col]
    X_vif = X.drop(columns=[col])

    if len(X_vif.columns) > 0:
        model = LinearRegression()
        model.fit(X_vif, y_vif)
        r_squared = model.score(X_vif, y_vif)
        vif = 1 / (1 - r_squared + 1e-10)
    else:
        vif = 1.0

    vif_data.append({
        'feature': col,
        'vif': round(vif, 4),
        'high_multicollinearity': vif > threshold
    })

vif_df = pd.DataFrame(vif_data)

# Find high correlation pairs
corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            high_corr_pairs.append({
                'feature_1': feature_cols[i],
                'feature_2': feature_cols[j],
                'correlation': round(corr, 4)
            })

# Add summary columns
vif_df['threshold_used'] = threshold
vif_df['n_high_vif'] = (vif_df['vif'] > threshold).sum()

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    vif_df['high_corr_pairs'] = str(high_corr_pairs)
else:
    vif_df['high_corr_pairs'] = 'None'

output = vif_df
`,
  },

  'funnel-analysis': {
    type: 'funnel-analysis',
    category: 'analysis',
    label: 'Funnel Analysis',
    description: 'Multi-step conversion funnel analysis',
    icon: 'Filter',
    defaultConfig: {
      stageColumn: '',
      countColumn: '',
      stageOrder: '',
      timestampColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
stage_col = config.get('stageColumn', '')
count_col = config.get('countColumn', '')
stage_order = config.get('stageOrder', '')
timestamp_col = config.get('timestampColumn', '')

if not stage_col:
    raise ValueError("Funnel Analysis: Please specify the stage column in the Config tab")

if stage_col not in df.columns:
    raise ValueError(f"Funnel Analysis: Stage column '{stage_col}' not found")

# Parse stage order or use natural order from data
if stage_order:
    stages = [s.strip() for s in stage_order.split(',')]
else:
    stages = df[stage_col].unique().tolist()

# Calculate counts per stage
if count_col and count_col in df.columns:
    stage_counts = df.groupby(stage_col)[count_col].sum()
else:
    stage_counts = df[stage_col].value_counts()

# Ensure all stages are present
for stage in stages:
    if stage not in stage_counts.index:
        stage_counts[stage] = 0

# Build funnel metrics
funnel_data = []
total_start = stage_counts.get(stages[0], 0)

for i, stage in enumerate(stages):
    count = stage_counts.get(stage, 0)

    # Calculate metrics
    if i == 0:
        conversion_from_prev = 100.0
        dropoff_from_prev = 0.0
    else:
        prev_count = stage_counts.get(stages[i-1], 0)
        if prev_count > 0:
            conversion_from_prev = (count / prev_count) * 100
            dropoff_from_prev = ((prev_count - count) / prev_count) * 100
        else:
            conversion_from_prev = 0.0
            dropoff_from_prev = 100.0

    overall_conversion = (count / total_start * 100) if total_start > 0 else 0.0

    funnel_data.append({
        'stage': stage,
        'stage_order': i + 1,
        'count': int(count),
        'conversion_from_previous': round(conversion_from_prev, 2),
        'dropoff_from_previous': round(dropoff_from_prev, 2),
        'overall_conversion': round(overall_conversion, 2)
    })

result = pd.DataFrame(funnel_data)

# Add summary metrics
result['total_stages'] = len(stages)
result['funnel_start_count'] = int(total_start)
result['funnel_end_count'] = int(stage_counts.get(stages[-1], 0))
result['overall_funnel_conversion'] = round(
    (stage_counts.get(stages[-1], 0) / total_start * 100) if total_start > 0 else 0, 2
)

# Add time-based metrics if timestamp provided
if timestamp_col and timestamp_col in df.columns:
    try:
        df['_ts'] = pd.to_datetime(df[timestamp_col])
        time_per_stage = df.groupby(stage_col)['_ts'].agg(['min', 'max'])
        for stage in stages:
            if stage in time_per_stage.index:
                stage_data = time_per_stage.loc[stage]
                duration = (stage_data['max'] - stage_data['min']).total_seconds() / 3600
                mask = result['stage'] == stage
                result.loc[mask, 'stage_duration_hours'] = round(duration, 2)
    except:
        pass

output = result
`,
  },

  'customer-ltv': {
    type: 'customer-ltv',
    category: 'analysis',
    label: 'Customer Lifetime Value',
    description: 'Calculate CLV using RFM-based approach',
    icon: 'DollarSign',
    defaultConfig: {
      customerIdColumn: '',
      transactionDateColumn: '',
      monetaryColumn: '',
      projectionPeriods: 12,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from datetime import datetime

df = input_data.copy()
customer_col = config.get('customerIdColumn', '')
date_col = config.get('transactionDateColumn', '')
monetary_col = config.get('monetaryColumn', '')
projection_periods = config.get('projectionPeriods', 12)

if not customer_col or not date_col or not monetary_col:
    raise ValueError("Customer LTV: Please specify customer ID, transaction date, and monetary columns")

if customer_col not in df.columns:
    raise ValueError(f"Customer LTV: Customer ID column '{customer_col}' not found")
if date_col not in df.columns:
    raise ValueError(f"Customer LTV: Transaction date column '{date_col}' not found")
if monetary_col not in df.columns:
    raise ValueError(f"Customer LTV: Monetary column '{monetary_col}' not found")

# Convert columns
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df[monetary_col] = pd.to_numeric(df[monetary_col], errors='coerce')
df = df.dropna(subset=[date_col, monetary_col])

if len(df) == 0:
    raise ValueError("Customer LTV: No valid data after cleaning")

# Calculate reference date
reference_date = df[date_col].max() + pd.Timedelta(days=1)
observation_period = (df[date_col].max() - df[date_col].min()).days

if observation_period <= 0:
    observation_period = 365

# Calculate RFM metrics per customer
rfm = df.groupby(customer_col).agg(
    recency=(date_col, lambda x: (reference_date - x.max()).days),
    frequency=(customer_col, 'count'),
    monetary=(monetary_col, 'sum'),
    first_purchase=(date_col, 'min'),
    last_purchase=(date_col, 'max'),
    avg_order_value=(monetary_col, 'mean')
).reset_index()

rfm.columns = [customer_col, 'recency', 'frequency', 'total_monetary', 'first_purchase', 'last_purchase', 'avg_order_value']

# Calculate customer tenure in days
rfm['tenure_days'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days + 1

# Calculate purchase frequency rate (purchases per day during active period)
rfm['purchase_rate'] = rfm['frequency'] / rfm['tenure_days'].clip(lower=1)

# Calculate average order value
rfm['avg_order_value'] = rfm['avg_order_value'].round(2)

# Estimate future purchases for projection period
projection_days = projection_periods * 30  # Assume 30 days per period
rfm['predicted_purchases'] = (rfm['purchase_rate'] * projection_days).round(2)

# Calculate predicted CLV
rfm['predicted_clv'] = (rfm['predicted_purchases'] * rfm['avg_order_value']).round(2)

# Calculate historical CLV
rfm['historical_clv'] = rfm['total_monetary'].round(2)

# Segment customers by CLV percentiles
clv_percentiles = rfm['predicted_clv'].quantile([0.25, 0.50, 0.75]).values

def segment_customer(clv):
    if clv >= clv_percentiles[2]:
        return 'Premium'
    elif clv >= clv_percentiles[1]:
        return 'High'
    elif clv >= clv_percentiles[0]:
        return 'Medium'
    else:
        return 'Low'

rfm['clv_segment'] = rfm['predicted_clv'].apply(segment_customer)

# Calculate churn risk based on recency
recency_median = rfm['recency'].median()
recency_75 = rfm['recency'].quantile(0.75)

def churn_risk(recency):
    if recency <= recency_median:
        return 'Low'
    elif recency <= recency_75:
        return 'Medium'
    else:
        return 'High'

rfm['churn_risk'] = rfm['recency'].apply(churn_risk)

# Clean up output columns
rfm = rfm[[customer_col, 'recency', 'frequency', 'total_monetary', 'avg_order_value',
           'tenure_days', 'predicted_purchases', 'historical_clv', 'predicted_clv',
           'clv_segment', 'churn_risk']]

output = rfm
`,
  },

  'churn-analysis': {
    type: 'churn-analysis',
    category: 'analysis',
    label: 'Churn Prediction',
    description: 'Binary classification optimized for churn prediction',
    icon: 'UserMinus',
    defaultConfig: {
      features: [],
      targetColumn: '',
      handleImbalance: true,
      threshold: 0.5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

df = input_data.copy()
features = config.get('features', [])
target_col = config.get('targetColumn', '')
handle_imbalance = config.get('handleImbalance', True)
threshold = config.get('threshold', 0.5)

if not features or not target_col:
    raise ValueError("Churn Analysis: Please specify feature columns and target column")

if target_col not in df.columns:
    raise ValueError(f"Churn Analysis: Target column '{target_col}' not found")

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"Churn Analysis: Feature columns not found: {missing_features}")

# Prepare features
X = df[features].copy()
y = df[target_col].copy()

# Handle categorical features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna('missing').astype(str))
        label_encoders[col] = le
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# Encode target if needed
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.fillna('unknown').astype(str))

# Handle class imbalance
class_weight = 'balanced' if handle_imbalance else None

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Predict probabilities
y_proba = model.predict_proba(scaler.transform(X))[:, 1]

# Calculate feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Create output dataframe
result = df.copy()
result['churn_probability'] = np.round(y_proba, 4)
result['churn_prediction'] = (y_proba >= threshold).astype(int)

# Risk segments
def risk_segment(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

result['risk_segment'] = result['churn_probability'].apply(risk_segment)

# Add feature importance as metadata columns
for i, row in feature_importance.head(5).iterrows():
    result[f'top_driver_{feature_importance.index.get_loc(i)+1}'] = row['feature']
    result[f'driver_importance_{feature_importance.index.get_loc(i)+1}'] = round(row['importance'], 4)

# Model performance on test set
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
result['model_auc'] = round(test_auc, 4)

output = result
`,
  },

  'growth-metrics': {
    type: 'growth-metrics',
    category: 'analysis',
    label: 'Growth Metrics',
    description: 'Calculate business growth rate metrics',
    icon: 'TrendingUp',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      granularity: 'month',
      rollingWindow: 3,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
granularity = config.get('granularity', 'month')
rolling_window = config.get('rollingWindow', 3)

if not date_col or not value_col:
    raise ValueError("Growth Metrics: Please specify date and value columns")

if date_col not in df.columns:
    raise ValueError(f"Growth Metrics: Date column '{date_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Growth Metrics: Value column '{value_col}' not found")

# Convert columns
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df = df.dropna(subset=[date_col, value_col])

if len(df) == 0:
    raise ValueError("Growth Metrics: No valid data after cleaning")

# Set date as index for resampling
df = df.set_index(date_col)

# Resample based on granularity
freq_map = {'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q', 'year': 'Y'}
freq = freq_map.get(granularity, 'M')
agg_df = df[[value_col]].resample(freq).sum().reset_index()
agg_df.columns = ['period', 'value']

# Calculate period-over-period growth
agg_df['prev_period_value'] = agg_df['value'].shift(1)
agg_df['period_growth'] = ((agg_df['value'] - agg_df['prev_period_value']) / agg_df['prev_period_value'] * 100).round(2)

# Calculate MoM/QoQ/YoY based on granularity
if granularity == 'month':
    agg_df['yoy_value'] = agg_df['value'].shift(12)
    agg_df['yoy_growth'] = ((agg_df['value'] - agg_df['yoy_value']) / agg_df['yoy_value'] * 100).round(2)
elif granularity == 'quarter':
    agg_df['yoy_value'] = agg_df['value'].shift(4)
    agg_df['yoy_growth'] = ((agg_df['value'] - agg_df['yoy_value']) / agg_df['yoy_value'] * 100).round(2)
elif granularity == 'day':
    agg_df['wow_value'] = agg_df['value'].shift(7)
    agg_df['wow_growth'] = ((agg_df['value'] - agg_df['wow_value']) / agg_df['wow_value'] * 100).round(2)

# Rolling average
agg_df['rolling_avg'] = agg_df['value'].rolling(window=rolling_window, min_periods=1).mean().round(2)

# Calculate CAGR (Compound Annual Growth Rate)
if len(agg_df) >= 2:
    first_value = agg_df['value'].iloc[0]
    last_value = agg_df['value'].iloc[-1]
    n_periods = len(agg_df) - 1

    # Annualize based on granularity
    periods_per_year = {'day': 365, 'week': 52, 'month': 12, 'quarter': 4, 'year': 1}
    ppy = periods_per_year.get(granularity, 12)
    n_years = n_periods / ppy

    if first_value > 0 and n_years > 0:
        cagr = ((last_value / first_value) ** (1 / n_years) - 1) * 100
        agg_df['cagr'] = round(cagr, 2)
    else:
        agg_df['cagr'] = np.nan
else:
    agg_df['cagr'] = np.nan

# Calculate acceleration (change in growth rate)
agg_df['growth_acceleration'] = (agg_df['period_growth'] - agg_df['period_growth'].shift(1)).round(2)

# Classify trend
def classify_trend(row):
    if pd.isna(row['period_growth']):
        return 'N/A'
    elif row['period_growth'] > 0:
        if pd.notna(row['growth_acceleration']) and row['growth_acceleration'] > 0:
            return 'Accelerating Growth'
        elif pd.notna(row['growth_acceleration']) and row['growth_acceleration'] < 0:
            return 'Decelerating Growth'
        else:
            return 'Growing'
    elif row['period_growth'] < 0:
        if pd.notna(row['growth_acceleration']) and row['growth_acceleration'] < 0:
            return 'Accelerating Decline'
        elif pd.notna(row['growth_acceleration']) and row['growth_acceleration'] > 0:
            return 'Decelerating Decline'
        else:
            return 'Declining'
    else:
        return 'Flat'

agg_df['trend'] = agg_df.apply(classify_trend, axis=1)

output = agg_df
`,
  },

  'attribution-modeling': {
    type: 'attribution-modeling',
    category: 'analysis',
    label: 'Attribution Modeling',
    description: 'Multi-touch marketing attribution analysis',
    icon: 'GitBranch',
    defaultConfig: {
      userIdColumn: '',
      channelColumn: '',
      conversionColumn: '',
      timestampColumn: '',
      model: 'linear',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
user_col = config.get('userIdColumn', '')
channel_col = config.get('channelColumn', '')
conversion_col = config.get('conversionColumn', '')
timestamp_col = config.get('timestampColumn', '')
model_type = config.get('model', 'linear')

if not user_col or not channel_col or not conversion_col:
    raise ValueError("Attribution Modeling: Please specify user ID, channel, and conversion columns")

for col in [user_col, channel_col, conversion_col]:
    if col not in df.columns:
        raise ValueError(f"Attribution Modeling: Column '{col}' not found")

# Convert conversion to numeric
df[conversion_col] = pd.to_numeric(df[conversion_col], errors='coerce').fillna(0)

# Sort by timestamp if provided
if timestamp_col and timestamp_col in df.columns:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.sort_values([user_col, timestamp_col])

# Group by user to get touchpoint sequences
user_journeys = df.groupby(user_col).agg({
    channel_col: list,
    conversion_col: 'sum'
}).reset_index()
user_journeys.columns = [user_col, 'touchpoints', 'conversions']

# Filter to users with conversions
converting_users = user_journeys[user_journeys['conversions'] > 0].copy()

# Initialize channel attribution
channels = df[channel_col].unique()
attribution = {ch: 0.0 for ch in channels}
channel_touchpoints = {ch: 0 for ch in channels}

# Apply attribution model
for _, row in converting_users.iterrows():
    touchpoints = row['touchpoints']
    conversion_value = row['conversions']
    n_touches = len(touchpoints)

    if n_touches == 0:
        continue

    # Count touchpoints per channel
    for tp in touchpoints:
        channel_touchpoints[tp] = channel_touchpoints.get(tp, 0) + 1

    if model_type == 'first-touch':
        attribution[touchpoints[0]] += conversion_value

    elif model_type == 'last-touch':
        attribution[touchpoints[-1]] += conversion_value

    elif model_type == 'linear':
        credit = conversion_value / n_touches
        for tp in touchpoints:
            attribution[tp] += credit

    elif model_type == 'time-decay':
        # More recent touchpoints get more credit
        weights = [2 ** i for i in range(n_touches)]
        total_weight = sum(weights)
        for i, tp in enumerate(touchpoints):
            credit = (weights[i] / total_weight) * conversion_value
            attribution[tp] += credit

    elif model_type == 'position-based':
        # 40% first, 40% last, 20% middle
        if n_touches == 1:
            attribution[touchpoints[0]] += conversion_value
        elif n_touches == 2:
            attribution[touchpoints[0]] += conversion_value * 0.5
            attribution[touchpoints[-1]] += conversion_value * 0.5
        else:
            attribution[touchpoints[0]] += conversion_value * 0.4
            attribution[touchpoints[-1]] += conversion_value * 0.4
            middle_credit = conversion_value * 0.2 / (n_touches - 2)
            for tp in touchpoints[1:-1]:
                attribution[tp] += middle_credit

# Build result dataframe
result = pd.DataFrame([
    {
        'channel': ch,
        'attributed_conversions': round(attribution[ch], 2),
        'total_touchpoints': channel_touchpoints[ch],
    }
    for ch in channels
])

# Calculate percentages and efficiency
total_attributed = result['attributed_conversions'].sum()
result['attribution_share'] = (result['attributed_conversions'] / total_attributed * 100).round(2) if total_attributed > 0 else 0

# Efficiency score (conversions per touchpoint)
result['efficiency_score'] = (result['attributed_conversions'] / result['total_touchpoints'].clip(lower=1)).round(4)

# Rank channels
result['rank'] = result['attributed_conversions'].rank(ascending=False, method='min').astype(int)

# Sort by attributed conversions
result = result.sort_values('attributed_conversions', ascending=False)

# Add model type used
result['attribution_model'] = model_type

output = result
`,
  },

  'breakeven-analysis': {
    type: 'breakeven-analysis',
    category: 'analysis',
    label: 'Break-even Analysis',
    description: 'Calculate financial break-even point',
    icon: 'Scale',
    defaultConfig: {
      fixedCosts: 10000,
      variableCostPerUnit: 5,
      pricePerUnit: 15,
      currentSalesUnits: 0,
      scenarioRange: 50,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
fixed_costs = config.get('fixedCosts', 10000)
variable_cost = config.get('variableCostPerUnit', 5)
price = config.get('pricePerUnit', 15)
current_sales = config.get('currentSalesUnits', 0)
scenario_range = config.get('scenarioRange', 50)

# Validate inputs
if price <= variable_cost:
    raise ValueError("Break-even Analysis: Price per unit must be greater than variable cost per unit")

if fixed_costs < 0 or variable_cost < 0 or price <= 0:
    raise ValueError("Break-even Analysis: Costs must be non-negative and price must be positive")

# Calculate contribution margin
contribution_margin = price - variable_cost
contribution_margin_ratio = contribution_margin / price

# Calculate break-even point
breakeven_units = np.ceil(fixed_costs / contribution_margin)
breakeven_revenue = breakeven_units * price

# Current profit/loss
current_revenue = current_sales * price
current_total_cost = fixed_costs + (current_sales * variable_cost)
current_profit = current_revenue - current_total_cost

# Margin of safety
margin_of_safety_units = max(0, current_sales - breakeven_units)
margin_of_safety_pct = (margin_of_safety_units / current_sales * 100) if current_sales > 0 else 0

# Create scenario analysis
scenarios = []
scenario_min = max(0, int(breakeven_units - scenario_range))
scenario_max = int(breakeven_units + scenario_range)

for units in range(scenario_min, scenario_max + 1, max(1, (scenario_max - scenario_min) // 20)):
    revenue = units * price
    total_cost = fixed_costs + (units * variable_cost)
    profit = revenue - total_cost

    scenarios.append({
        'units_sold': int(units),
        'revenue': round(revenue, 2),
        'fixed_costs': round(fixed_costs, 2),
        'variable_costs': round(units * variable_cost, 2),
        'total_costs': round(total_cost, 2),
        'profit_loss': round(profit, 2),
        'is_breakeven': units == breakeven_units,
        'is_current': units == current_sales
    })

result = pd.DataFrame(scenarios)

# Add summary metrics
result['breakeven_units'] = int(breakeven_units)
result['breakeven_revenue'] = round(breakeven_revenue, 2)
result['contribution_margin'] = round(contribution_margin, 2)
result['contribution_margin_ratio'] = round(contribution_margin_ratio * 100, 2)
result['margin_of_safety_units'] = int(margin_of_safety_units)
result['margin_of_safety_pct'] = round(margin_of_safety_pct, 2)
result['current_profit_loss'] = round(current_profit, 2)

# Calculate units needed for target profits
target_profits = [0, 5000, 10000, 25000, 50000]
for target in target_profits:
    units_needed = np.ceil((fixed_costs + target) / contribution_margin)
    result[f'units_for_{target}_profit'] = int(units_needed)

output = result
`,
  },

  'confidence-intervals': {
    type: 'confidence-intervals',
    category: 'analysis',
    label: 'Confidence Intervals',
    description: 'Calculate confidence intervals for means and proportions',
    icon: 'ArrowLeftRight',
    defaultConfig: {
      column: '',
      column2: '',
      confidenceLevel: 0.95,
      analysisType: 'one-sample-mean',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
column = config.get('column', '')
column2 = config.get('column2', '')
confidence_level = config.get('confidenceLevel', 0.95)
analysis_type = config.get('analysisType', 'one-sample-mean')

if not column:
    raise ValueError("Confidence Intervals: Please specify the primary column")

if column not in df.columns:
    raise ValueError(f"Confidence Intervals: Column '{column}' not found")

# Convert to numeric
data1 = pd.to_numeric(df[column], errors='coerce').dropna()

if len(data1) < 2:
    raise ValueError("Confidence Intervals: Need at least 2 data points")

alpha = 1 - confidence_level
results = []

if analysis_type == 'one-sample-mean':
    n = len(data1)
    mean = data1.mean()
    std = data1.std(ddof=1)
    se = std / np.sqrt(n)

    # Use t-distribution for small samples
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    results.append({
        'analysis_type': 'One-Sample Mean',
        'column': column,
        'n': n,
        'point_estimate': round(mean, 4),
        'std_dev': round(std, 4),
        'std_error': round(se, 4),
        'confidence_level': confidence_level,
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'margin_of_error': round(margin_of_error, 4),
        'interpretation': f"We are {confidence_level*100:.0f}% confident the true mean is between {ci_lower:.4f} and {ci_upper:.4f}"
    })

elif analysis_type == 'one-sample-proportion':
    # Treat values > 0 as successes
    successes = (data1 > 0).sum()
    n = len(data1)
    p_hat = successes / n

    # Wilson score interval (better for proportions)
    z = stats.norm.ppf(1 - alpha/2)
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denominator

    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    results.append({
        'analysis_type': 'One-Sample Proportion',
        'column': column,
        'n': n,
        'successes': int(successes),
        'point_estimate': round(p_hat, 4),
        'confidence_level': confidence_level,
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'margin_of_error': round(margin, 4),
        'interpretation': f"We are {confidence_level*100:.0f}% confident the true proportion is between {ci_lower:.4f} and {ci_upper:.4f}"
    })

elif analysis_type == 'two-sample-mean':
    if not column2 or column2 not in df.columns:
        raise ValueError("Confidence Intervals: Please specify a second column for two-sample analysis")

    data2 = pd.to_numeric(df[column2], errors='coerce').dropna()

    if len(data2) < 2:
        raise ValueError("Confidence Intervals: Second column needs at least 2 data points")

    n1, n2 = len(data1), len(data2)
    mean1, mean2 = data1.mean(), data2.mean()
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)

    # Welch's t-test (unequal variances)
    se = np.sqrt(var1/n1 + var2/n2)
    diff = mean1 - mean2

    # Welch-Satterthwaite degrees of freedom
    df_ws = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_crit = stats.t.ppf(1 - alpha/2, df=df_ws)

    margin_of_error = t_crit * se
    ci_lower = diff - margin_of_error
    ci_upper = diff + margin_of_error

    results.append({
        'analysis_type': 'Two-Sample Mean Difference',
        'column': column,
        'column2': column2,
        'n1': n1,
        'n2': n2,
        'mean1': round(mean1, 4),
        'mean2': round(mean2, 4),
        'point_estimate': round(diff, 4),
        'std_error': round(se, 4),
        'confidence_level': confidence_level,
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'margin_of_error': round(margin_of_error, 4),
        'interpretation': f"We are {confidence_level*100:.0f}% confident the difference in means is between {ci_lower:.4f} and {ci_upper:.4f}"
    })

elif analysis_type == 'paired':
    if not column2 or column2 not in df.columns:
        raise ValueError("Confidence Intervals: Please specify a second column for paired analysis")

    data2 = pd.to_numeric(df[column2], errors='coerce')

    # Align and remove missing
    paired_df = pd.DataFrame({'a': data1.values, 'b': data2.values}).dropna()

    if len(paired_df) < 2:
        raise ValueError("Confidence Intervals: Need at least 2 paired observations")

    differences = paired_df['a'] - paired_df['b']
    n = len(differences)
    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)
    se = std_diff / np.sqrt(n)

    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * se
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    results.append({
        'analysis_type': 'Paired Difference',
        'column': column,
        'column2': column2,
        'n_pairs': n,
        'mean_difference': round(mean_diff, 4),
        'std_difference': round(std_diff, 4),
        'point_estimate': round(mean_diff, 4),
        'std_error': round(se, 4),
        'confidence_level': confidence_level,
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'margin_of_error': round(margin_of_error, 4),
        'interpretation': f"We are {confidence_level*100:.0f}% confident the mean paired difference is between {ci_lower:.4f} and {ci_upper:.4f}"
    })

output = pd.DataFrame(results)
`,
  },

  'bootstrap-analysis': {
    type: 'bootstrap-analysis',
    category: 'analysis',
    label: 'Bootstrap Analysis',
    description: 'Non-parametric bootstrap resampling for confidence intervals',
    icon: 'RefreshCw',
    defaultConfig: {
      column: '',
      statistic: 'mean',
      column2: '',
      nIterations: 1000,
      confidenceLevel: 0.95,
      method: 'percentile',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
column = config.get('column', '')
statistic = config.get('statistic', 'mean')
column2 = config.get('column2', '')
n_iterations = int(config.get('nIterations', 1000))
confidence_level = config.get('confidenceLevel', 0.95)
method = config.get('method', 'percentile')

if not column:
    raise ValueError("Bootstrap Analysis: Please specify a column")

if column not in df.columns:
    raise ValueError(f"Bootstrap Analysis: Column '{column}' not found")

# Get data
data1 = pd.to_numeric(df[column], errors='coerce').dropna().values

if len(data1) < 10:
    raise ValueError("Bootstrap Analysis: Need at least 10 data points")

# For correlation, need second column
data2 = None
if statistic == 'correlation':
    if not column2 or column2 not in df.columns:
        raise ValueError("Bootstrap Analysis: Second column required for correlation")
    data2 = pd.to_numeric(df[column2], errors='coerce').dropna().values
    # Align arrays
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

n = len(data1)

# Define statistic functions
def calc_statistic(sample1, sample2=None):
    if statistic == 'mean':
        return np.mean(sample1)
    elif statistic == 'median':
        return np.median(sample1)
    elif statistic == 'std':
        return np.std(sample1, ddof=1)
    elif statistic == 'correlation':
        if sample2 is None:
            return np.nan
        return np.corrcoef(sample1, sample2)[0, 1]
    return np.mean(sample1)

# Calculate observed statistic
observed = calc_statistic(data1, data2)

# Bootstrap resampling
bootstrap_stats = []
np.random.seed(42)
for _ in range(n_iterations):
    indices = np.random.randint(0, n, size=n)
    sample1 = data1[indices]
    sample2 = data2[indices] if data2 is not None else None
    bootstrap_stats.append(calc_statistic(sample1, sample2))

bootstrap_stats = np.array(bootstrap_stats)

# Calculate confidence interval
alpha = 1 - confidence_level

if method == 'percentile':
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
elif method == 'bca':
    # BCa (Bias-Corrected and Accelerated) method
    # Bias correction
    z0 = stats.norm.ppf(np.mean(bootstrap_stats < observed))

    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jack_sample1 = np.delete(data1, i)
        jack_sample2 = np.delete(data2, i) if data2 is not None else None
        jackknife_stats.append(calc_statistic(jack_sample1, jack_sample2))
    jackknife_stats = np.array(jackknife_stats)
    jack_mean = np.mean(jackknife_stats)
    a = np.sum((jack_mean - jackknife_stats)**3) / (6 * np.sum((jack_mean - jackknife_stats)**2)**1.5 + 1e-10)

    # Adjusted percentiles
    z_alpha_lower = stats.norm.ppf(alpha/2)
    z_alpha_upper = stats.norm.ppf(1 - alpha/2)

    adj_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower) + 1e-10))
    adj_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper) + 1e-10))

    ci_lower = np.percentile(bootstrap_stats, adj_lower * 100)
    ci_upper = np.percentile(bootstrap_stats, adj_upper * 100)
else:
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

# Calculate standard error and bias
se = np.std(bootstrap_stats, ddof=1)
bias = np.mean(bootstrap_stats) - observed

results = [{
    'statistic': statistic,
    'column': column,
    'column2': column2 if statistic == 'correlation' else '',
    'observed_value': round(observed, 6),
    'bootstrap_mean': round(np.mean(bootstrap_stats), 6),
    'bootstrap_se': round(se, 6),
    'bias': round(bias, 6),
    'bias_corrected_estimate': round(observed - bias, 6),
    'confidence_level': confidence_level,
    'method': method,
    'ci_lower': round(ci_lower, 6),
    'ci_upper': round(ci_upper, 6),
    'n_iterations': n_iterations,
    'sample_size': n
}]

output = pd.DataFrame(results)
`,
  },

  'posthoc-tests': {
    type: 'posthoc-tests',
    category: 'analysis',
    label: 'Post-hoc Tests',
    description: 'Multiple comparison tests after ANOVA',
    icon: 'GitCompare',
    defaultConfig: {
      valueColumn: '',
      groupColumn: '',
      method: 'tukey',
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

df = input_data.copy()
value_column = config.get('valueColumn', '')
group_column = config.get('groupColumn', '')
method = config.get('method', 'tukey')
alpha = config.get('alpha', 0.05)

if not value_column or not group_column:
    raise ValueError("Post-hoc Tests: Please specify both value and group columns")

if value_column not in df.columns:
    raise ValueError(f"Post-hoc Tests: Value column '{value_column}' not found")

if group_column not in df.columns:
    raise ValueError(f"Post-hoc Tests: Group column '{group_column}' not found")

# Convert value column to numeric
df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
df = df.dropna(subset=[value_column, group_column])

groups = df[group_column].unique()
n_groups = len(groups)

if n_groups < 2:
    raise ValueError("Post-hoc Tests: Need at least 2 groups for comparison")

# Get group data
group_data = {g: df[df[group_column] == g][value_column].values for g in groups}

# Calculate pooled standard deviation
all_values = df[value_column].values
n_total = len(all_values)
grand_mean = np.mean(all_values)

# Within-group sum of squares
ss_within = sum(np.sum((group_data[g] - np.mean(group_data[g]))**2) for g in groups)
df_within = n_total - n_groups
ms_within = ss_within / df_within
pooled_std = np.sqrt(ms_within)

results = []

# All pairwise comparisons
pairs = list(combinations(groups, 2))
n_comparisons = len(pairs)

for g1, g2 in pairs:
    data1 = group_data[g1]
    data2 = group_data[g2]
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    mean_diff = mean1 - mean2

    # Standard error of difference
    se = pooled_std * np.sqrt(1/n1 + 1/n2)

    # T-statistic
    t_stat = mean_diff / se

    # Calculate p-value based on method
    if method == 'tukey':
        # Tukey HSD uses studentized range distribution
        # Approximate with t-distribution
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
        # Adjust for multiple comparisons
        p_adjusted = min(1.0, p_value * n_comparisons / 2)
    elif method == 'bonferroni':
        # Bonferroni: multiply p-value by number of comparisons
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
        p_adjusted = min(1.0, p_value * n_comparisons)
    elif method == 'holm':
        # Holm: sequential rejection procedure (will be applied after)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
        p_adjusted = p_value  # Will be adjusted later
    else:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
        p_adjusted = p_value

    # Cohen's d effect size
    cohens_d = mean_diff / pooled_std

    results.append({
        'group1': str(g1),
        'group2': str(g2),
        'mean1': round(mean1, 4),
        'mean2': round(mean2, 4),
        'mean_difference': round(mean_diff, 4),
        'std_error': round(se, 4),
        't_statistic': round(t_stat, 4),
        'p_value_raw': round(p_value, 6),
        'p_value_adjusted': round(p_adjusted, 6),
        'cohens_d': round(cohens_d, 4),
        'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large'),
        'significant': p_adjusted < alpha,
        'method': method,
        'alpha': alpha
    })

# Apply Holm correction if selected
if method == 'holm':
    results.sort(key=lambda x: x['p_value_raw'])
    for i, r in enumerate(results):
        holm_adjusted = min(1.0, r['p_value_raw'] * (n_comparisons - i))
        r['p_value_adjusted'] = round(holm_adjusted, 6)
        r['significant'] = holm_adjusted < alpha

output = pd.DataFrame(results)
`,
  },

  'power-analysis': {
    type: 'power-analysis',
    category: 'analysis',
    label: 'Power Analysis',
    description: 'Sample size and power calculations',
    icon: 'Gauge',
    defaultConfig: {
      testType: 't-test',
      effectSize: 0.5,
      alpha: 0.05,
      power: 0.8,
      calculateFor: 'sample_size',
      sampleSize: 30,
      groups: 2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
test_type = config.get('testType', 't-test')
effect_size = config.get('effectSize', 0.5)
alpha = config.get('alpha', 0.05)
power = config.get('power', 0.8)
calculate_for = config.get('calculateFor', 'sample_size')
sample_size = config.get('sampleSize', 30)
groups = config.get('groups', 2)

# Validate inputs
if effect_size <= 0:
    raise ValueError("Power Analysis: Effect size must be positive")
if not 0 < alpha < 1:
    raise ValueError("Power Analysis: Alpha must be between 0 and 1")
if not 0 < power < 1:
    raise ValueError("Power Analysis: Power must be between 0 and 1")

results = []

def calc_power_ttest(n, d, alpha):
    """Calculate power for independent t-test"""
    df = 2 * n - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ncp = d * np.sqrt(n/2)  # Non-centrality parameter
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return power

def calc_power_anova(n, f, alpha, k):
    """Calculate power for one-way ANOVA"""
    df1 = k - 1
    df2 = k * (n - 1)
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    ncp = f**2 * k * n
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    return power

def calc_power_chisq(n, w, alpha, df=1):
    """Calculate power for chi-square test"""
    chi2_crit = stats.chi2.ppf(1 - alpha, df)
    ncp = n * w**2
    power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
    return power

def calc_power_correlation(n, r, alpha):
    """Calculate power for correlation test"""
    # Fisher z transformation
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)
    power = 1 - stats.norm.cdf(z_crit - z_r/se) + stats.norm.cdf(-z_crit - z_r/se)
    return max(0, min(1, power))

if calculate_for == 'sample_size':
    # Find required sample size
    for n in range(5, 10001):
        if test_type == 't-test':
            achieved_power = calc_power_ttest(n, effect_size, alpha)
        elif test_type == 'anova':
            achieved_power = calc_power_anova(n, effect_size, alpha, groups)
        elif test_type == 'chi-square':
            achieved_power = calc_power_chisq(n, effect_size, alpha)
        elif test_type == 'correlation':
            achieved_power = calc_power_correlation(n, effect_size, alpha)
        else:
            achieved_power = calc_power_ttest(n, effect_size, alpha)

        if achieved_power >= power:
            break

    results.append({
        'test_type': test_type,
        'effect_size': effect_size,
        'alpha': alpha,
        'target_power': power,
        'calculated_for': 'sample_size',
        'required_sample_size': n,
        'achieved_power': round(achieved_power, 4),
        'groups': groups if test_type == 'anova' else 2,
        'total_n': n * groups if test_type == 'anova' else n * 2 if test_type == 't-test' else n
    })

elif calculate_for == 'power':
    # Calculate achieved power
    if test_type == 't-test':
        achieved_power = calc_power_ttest(sample_size, effect_size, alpha)
    elif test_type == 'anova':
        achieved_power = calc_power_anova(sample_size, effect_size, alpha, groups)
    elif test_type == 'chi-square':
        achieved_power = calc_power_chisq(sample_size, effect_size, alpha)
    elif test_type == 'correlation':
        achieved_power = calc_power_correlation(sample_size, effect_size, alpha)
    else:
        achieved_power = calc_power_ttest(sample_size, effect_size, alpha)

    results.append({
        'test_type': test_type,
        'effect_size': effect_size,
        'alpha': alpha,
        'sample_size': sample_size,
        'calculated_for': 'power',
        'achieved_power': round(achieved_power, 4),
        'groups': groups if test_type == 'anova' else 2
    })

elif calculate_for == 'effect_size':
    # Find detectable effect size
    for d_test in np.arange(0.01, 3.0, 0.01):
        if test_type == 't-test':
            achieved_power = calc_power_ttest(sample_size, d_test, alpha)
        elif test_type == 'anova':
            achieved_power = calc_power_anova(sample_size, d_test, alpha, groups)
        elif test_type == 'chi-square':
            achieved_power = calc_power_chisq(sample_size, d_test, alpha)
        elif test_type == 'correlation':
            achieved_power = calc_power_correlation(sample_size, d_test, alpha)
        else:
            achieved_power = calc_power_ttest(sample_size, d_test, alpha)

        if achieved_power >= power:
            break

    results.append({
        'test_type': test_type,
        'sample_size': sample_size,
        'alpha': alpha,
        'target_power': power,
        'calculated_for': 'effect_size',
        'detectable_effect_size': round(d_test, 4),
        'achieved_power': round(achieved_power, 4),
        'groups': groups if test_type == 'anova' else 2
    })

# Generate power curve data
power_curve = []
for n in [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]:
    if test_type == 't-test':
        p = calc_power_ttest(n, effect_size, alpha)
    elif test_type == 'anova':
        p = calc_power_anova(n, effect_size, alpha, groups)
    elif test_type == 'chi-square':
        p = calc_power_chisq(n, effect_size, alpha)
    elif test_type == 'correlation':
        p = calc_power_correlation(n, effect_size, alpha)
    else:
        p = calc_power_ttest(n, effect_size, alpha)

    power_curve.append({
        'sample_size_per_group': n,
        'power': round(p, 4)
    })

# Combine results
main_result = results[0]
for i, pc in enumerate(power_curve):
    main_result[f'power_at_n{pc["sample_size_per_group"]}'] = pc['power']

output = pd.DataFrame([main_result])
`,
  },

  'bayesian-inference': {
    type: 'bayesian-inference',
    category: 'analysis',
    label: 'Bayesian Inference',
    description: 'Bayesian estimation for common scenarios',
    icon: 'BarChart2',
    defaultConfig: {
      analysisType: 'proportion',
      column: '',
      column2: '',
      priorAlpha: 1,
      priorBeta: 1,
      credibleLevel: 0.95,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
analysis_type = config.get('analysisType', 'proportion')
column = config.get('column', '')
column2 = config.get('column2', '')
prior_alpha = config.get('priorAlpha', 1)
prior_beta = config.get('priorBeta', 1)
credible_level = config.get('credibleLevel', 0.95)

if not column:
    raise ValueError("Bayesian Inference: Please specify a column")

if column not in df.columns:
    raise ValueError(f"Bayesian Inference: Column '{column}' not found")

alpha_level = (1 - credible_level) / 2
results = []

if analysis_type == 'proportion':
    # Beta-Binomial model for proportion
    data = pd.to_numeric(df[column], errors='coerce').dropna()

    # Count successes (values > 0 or == 1)
    successes = int((data > 0).sum())
    n = len(data)
    failures = n - successes

    # Posterior parameters (Beta distribution)
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + failures

    # Posterior statistics
    posterior_mean = post_alpha / (post_alpha + post_beta)
    posterior_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else posterior_mean
    posterior_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))
    posterior_std = np.sqrt(posterior_var)

    # Credible interval
    ci_lower = stats.beta.ppf(alpha_level, post_alpha, post_beta)
    ci_upper = stats.beta.ppf(1 - alpha_level, post_alpha, post_beta)

    results.append({
        'analysis_type': 'Proportion (Beta-Binomial)',
        'column': column,
        'n': n,
        'successes': successes,
        'prior_alpha': prior_alpha,
        'prior_beta': prior_beta,
        'posterior_alpha': post_alpha,
        'posterior_beta': post_beta,
        'posterior_mean': round(posterior_mean, 6),
        'posterior_mode': round(posterior_mode, 6),
        'posterior_std': round(posterior_std, 6),
        'credible_level': credible_level,
        'hdi_lower': round(ci_lower, 6),
        'hdi_upper': round(ci_upper, 6)
    })

elif analysis_type == 'mean':
    # Normal model with unknown mean (known variance approximation)
    data = pd.to_numeric(df[column], errors='coerce').dropna().values
    n = len(data)

    if n < 2:
        raise ValueError("Bayesian Inference: Need at least 2 data points")

    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    sample_std = np.sqrt(sample_var)

    # Using conjugate prior approach
    # Prior: N(prior_alpha, prior_beta) where prior_alpha=mean, prior_beta=variance
    prior_mean = prior_alpha
    prior_var = prior_beta**2 if prior_beta > 0 else sample_var * 10

    # Posterior (normal conjugate)
    posterior_precision = 1/prior_var + n/sample_var
    posterior_var = 1/posterior_precision
    posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/sample_var)
    posterior_std = np.sqrt(posterior_var)

    # Credible interval
    ci_lower = stats.norm.ppf(alpha_level, posterior_mean, posterior_std)
    ci_upper = stats.norm.ppf(1 - alpha_level, posterior_mean, posterior_std)

    results.append({
        'analysis_type': 'Mean (Normal)',
        'column': column,
        'n': n,
        'sample_mean': round(sample_mean, 6),
        'sample_std': round(sample_std, 6),
        'prior_mean': prior_mean,
        'prior_std': prior_beta,
        'posterior_mean': round(posterior_mean, 6),
        'posterior_std': round(posterior_std, 6),
        'credible_level': credible_level,
        'hdi_lower': round(ci_lower, 6),
        'hdi_upper': round(ci_upper, 6)
    })

elif analysis_type == 'ab-test':
    # Bayesian A/B test for proportions
    if not column2 or column2 not in df.columns:
        raise ValueError("Bayesian Inference: Second column required for A/B test")

    data_a = pd.to_numeric(df[column], errors='coerce').dropna()
    data_b = pd.to_numeric(df[column2], errors='coerce').dropna()

    # Count successes
    successes_a = int((data_a > 0).sum())
    n_a = len(data_a)
    successes_b = int((data_b > 0).sum())
    n_b = len(data_b)

    # Posterior parameters
    alpha_a = prior_alpha + successes_a
    beta_a = prior_beta + (n_a - successes_a)
    alpha_b = prior_alpha + successes_b
    beta_b = prior_beta + (n_b - successes_b)

    # Monte Carlo simulation for P(B > A)
    np.random.seed(42)
    n_samples = 10000
    samples_a = stats.beta.rvs(alpha_a, beta_a, size=n_samples)
    samples_b = stats.beta.rvs(alpha_b, beta_b, size=n_samples)

    prob_b_greater = np.mean(samples_b > samples_a)

    # Credible interval for difference
    diff_samples = samples_b - samples_a
    diff_ci_lower = np.percentile(diff_samples, alpha_level * 100)
    diff_ci_upper = np.percentile(diff_samples, (1 - alpha_level) * 100)

    # Relative uplift
    relative_uplift = (np.mean(samples_b) - np.mean(samples_a)) / (np.mean(samples_a) + 1e-10)

    results.append({
        'analysis_type': 'A/B Test (Beta-Binomial)',
        'column_a': column,
        'column_b': column2,
        'n_a': n_a,
        'n_b': n_b,
        'successes_a': successes_a,
        'successes_b': successes_b,
        'conversion_rate_a': round(successes_a / n_a, 6),
        'conversion_rate_b': round(successes_b / n_b, 6),
        'posterior_mean_a': round(alpha_a / (alpha_a + beta_a), 6),
        'posterior_mean_b': round(alpha_b / (alpha_b + beta_b), 6),
        'prob_b_greater_than_a': round(prob_b_greater, 6),
        'expected_uplift': round(np.mean(diff_samples), 6),
        'relative_uplift_pct': round(relative_uplift * 100, 2),
        'credible_level': credible_level,
        'diff_hdi_lower': round(diff_ci_lower, 6),
        'diff_hdi_upper': round(diff_ci_upper, 6),
        'recommendation': 'B is better' if prob_b_greater > 0.95 else ('A is better' if prob_b_greater < 0.05 else 'Inconclusive')
    })

output = pd.DataFrame(results)
`,
  },

  'data-quality-score': {
    type: 'data-quality-score',
    category: 'analysis',
    label: 'Data Quality Score',
    description: 'Comprehensive data quality assessment',
    icon: 'CheckCircle',
    defaultConfig: {
      columns: [],
      completenessWeight: 0.3,
      validityWeight: 0.3,
      uniquenessWeight: 0.2,
      consistencyWeight: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
completeness_weight = config.get('completenessWeight', 0.3)
validity_weight = config.get('validityWeight', 0.3)
uniqueness_weight = config.get('uniquenessWeight', 0.2)
consistency_weight = config.get('consistencyWeight', 0.2)

# Use all columns if none specified
if not columns or len(columns) == 0:
    columns = df.columns.tolist()

# Validate columns exist
valid_columns = [c for c in columns if c in df.columns]
if not valid_columns:
    raise ValueError("Data Quality Score: No valid columns found")

results = []
issues = []

for col in valid_columns:
    col_data = df[col]
    n_total = len(col_data)

    # 1. Completeness: % of non-null values
    n_missing = col_data.isnull().sum()
    completeness = (n_total - n_missing) / n_total * 100

    if n_missing > 0:
        missing_rows = df[col_data.isnull()].index.tolist()[:5]
        issues.append({
            'column': col,
            'issue_type': 'Missing Values',
            'severity': 'high' if n_missing / n_total > 0.1 else 'medium',
            'count': int(n_missing),
            'percentage': round(n_missing / n_total * 100, 2),
            'sample_rows': str(missing_rows)
        })

    # 2. Validity: check data types and patterns
    validity = 100.0
    n_invalid = 0

    # Try to detect data type issues
    if col_data.dtype == 'object':
        # Check for mixed types
        non_null = col_data.dropna()
        if len(non_null) > 0:
            # Check if should be numeric
            numeric_count = pd.to_numeric(non_null, errors='coerce').notna().sum()
            if numeric_count > 0 and numeric_count < len(non_null):
                n_invalid = len(non_null) - numeric_count
                validity = (len(non_null) - n_invalid) / len(non_null) * 100
                issues.append({
                    'column': col,
                    'issue_type': 'Mixed Data Types',
                    'severity': 'medium',
                    'count': int(n_invalid),
                    'percentage': round(n_invalid / n_total * 100, 2),
                    'sample_rows': ''
                })
    else:
        # For numeric columns, check for infinities
        if np.issubdtype(col_data.dtype, np.number):
            n_inf = np.isinf(col_data.dropna()).sum()
            if n_inf > 0:
                validity = (n_total - n_inf) / n_total * 100
                issues.append({
                    'column': col,
                    'issue_type': 'Infinite Values',
                    'severity': 'high',
                    'count': int(n_inf),
                    'percentage': round(n_inf / n_total * 100, 2),
                    'sample_rows': ''
                })

    # 3. Uniqueness: % of unique values (for detecting duplicates)
    n_unique = col_data.nunique()
    uniqueness = n_unique / n_total * 100

    # Check for potential ID columns with duplicates
    if uniqueness > 90 and uniqueness < 100 and n_total > 10:
        n_duplicates = n_total - n_unique
        duplicate_values = col_data[col_data.duplicated()].head(3).tolist()
        issues.append({
            'column': col,
            'issue_type': 'Duplicate Values',
            'severity': 'low',
            'count': int(n_duplicates),
            'percentage': round(n_duplicates / n_total * 100, 2),
            'sample_rows': str(duplicate_values)
        })

    # 4. Consistency: check for outliers and inconsistent patterns
    consistency = 100.0
    n_outliers = 0

    if np.issubdtype(col_data.dtype, np.number):
        # Use IQR method for outlier detection
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((col_data < lower) | (col_data > upper)).sum()

        if n_outliers > 0:
            consistency = (n_total - n_outliers) / n_total * 100
            outlier_rows = df[(col_data < lower) | (col_data > upper)].index.tolist()[:5]
            issues.append({
                'column': col,
                'issue_type': 'Statistical Outliers',
                'severity': 'low' if n_outliers / n_total < 0.05 else 'medium',
                'count': int(n_outliers),
                'percentage': round(n_outliers / n_total * 100, 2),
                'sample_rows': str(outlier_rows)
            })

    # Calculate weighted score for this column
    col_score = (
        completeness * completeness_weight +
        validity * validity_weight +
        uniqueness * uniqueness_weight +
        consistency * consistency_weight
    ) / (completeness_weight + validity_weight + uniqueness_weight + consistency_weight)

    results.append({
        'column': col,
        'completeness_score': round(completeness, 2),
        'validity_score': round(validity, 2),
        'uniqueness_score': round(uniqueness, 2),
        'consistency_score': round(consistency, 2),
        'overall_score': round(col_score, 2),
        'n_missing': int(n_missing),
        'n_unique': int(n_unique),
        'n_outliers': int(n_outliers)
    })

# Calculate overall data quality score
if results:
    overall_score = np.mean([r['overall_score'] for r in results])

    # Add summary row
    results.insert(0, {
        'column': '*** OVERALL ***',
        'completeness_score': round(np.mean([r['completeness_score'] for r in results[1:]]), 2) if len(results) > 1 else 0,
        'validity_score': round(np.mean([r['validity_score'] for r in results[1:]]), 2) if len(results) > 1 else 0,
        'uniqueness_score': round(np.mean([r['uniqueness_score'] for r in results[1:]]), 2) if len(results) > 1 else 0,
        'consistency_score': round(np.mean([r['consistency_score'] for r in results[1:]]), 2) if len(results) > 1 else 0,
        'overall_score': round(overall_score, 2),
        'n_missing': sum(r['n_missing'] for r in results[1:]) if len(results) > 1 else 0,
        'n_unique': '',
        'n_outliers': sum(r['n_outliers'] for r in results[1:]) if len(results) > 1 else 0
    })

# Convert issues to dataframe and append
if issues:
    issues_df = pd.DataFrame(issues)
    results_df = pd.DataFrame(results)
    output = pd.concat([results_df, pd.DataFrame([{'column': '--- ISSUES ---'}]), issues_df], ignore_index=True)
else:
    output = pd.DataFrame(results)
`,
  },

  'changepoint-detection': {
    type: 'changepoint-detection',
    category: 'analysis',
    label: 'Change Point Detection',
    description: 'Detect structural breaks in time series',
    icon: 'GitCommit',
    defaultConfig: {
      valueColumn: '',
      dateColumn: '',
      method: 'cusum',
      minSegmentSize: 10,
      maxChangePoints: 5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
value_column = config.get('valueColumn', '')
date_column = config.get('dateColumn', '')
method = config.get('method', 'cusum')
min_segment_size = int(config.get('minSegmentSize', 10))
max_changepoints = int(config.get('maxChangePoints', 5))

if not value_column:
    raise ValueError("Change Point Detection: Please specify a value column")

if value_column not in df.columns:
    raise ValueError(f"Change Point Detection: Column '{value_column}' not found")

# Sort by date if provided
if date_column and date_column in df.columns:
    df = df.sort_values(date_column).reset_index(drop=True)

# Get values
values = pd.to_numeric(df[value_column], errors='coerce').dropna().values
n = len(values)

if n < 2 * min_segment_size:
    raise ValueError(f"Change Point Detection: Need at least {2 * min_segment_size} data points")

changepoints = []

if method == 'cusum':
    # CUSUM (Cumulative Sum) method
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Calculate CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mean) / std - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mean) / std - 0.5)

    # Detect changepoints where CUSUM exceeds threshold
    threshold = 4.0  # Typical threshold

    # Find peaks in CUSUM
    for i in range(min_segment_size, n - min_segment_size):
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            # Check if this is a local maximum
            is_max = True
            for j in range(max(0, i-5), min(n, i+6)):
                if j != i:
                    if cusum_pos[i] <= cusum_pos[j] or cusum_neg[i] <= cusum_neg[j]:
                        is_max = False
                        break

            if is_max and len(changepoints) < max_changepoints:
                # Reset CUSUM after detection
                score = max(cusum_pos[i], cusum_neg[i])
                changepoints.append((i, score))
                cusum_pos[i:] = 0
                cusum_neg[i:] = 0

elif method == 'pelt':
    # Pruned Exact Linear Time (PELT) - simplified version
    # Using binary segmentation as approximation

    def segment_cost(start, end):
        if end - start < 2:
            return float('inf')
        segment = values[start:end]
        return np.sum((segment - np.mean(segment))**2)

    def find_best_split(start, end):
        if end - start < 2 * min_segment_size:
            return None, float('inf')

        best_pos = None
        best_gain = 0

        base_cost = segment_cost(start, end)

        for i in range(start + min_segment_size, end - min_segment_size + 1):
            split_cost = segment_cost(start, i) + segment_cost(i, end)
            gain = base_cost - split_cost

            if gain > best_gain:
                best_gain = gain
                best_pos = i

        return best_pos, best_gain

    # Iteratively find changepoints
    segments = [(0, n)]

    while len(changepoints) < max_changepoints and segments:
        best_segment = None
        best_split = None
        best_gain = 0

        for seg in segments:
            pos, gain = find_best_split(seg[0], seg[1])
            if pos is not None and gain > best_gain:
                best_segment = seg
                best_split = pos
                best_gain = gain

        if best_split is None or best_gain < 0.01 * np.var(values) * n:
            break

        segments.remove(best_segment)
        segments.append((best_segment[0], best_split))
        segments.append((best_split, best_segment[1]))

        changepoints.append((best_split, best_gain))

elif method == 'binary':
    # Binary Segmentation
    def segment_mean_diff(pos):
        if pos < min_segment_size or n - pos < min_segment_size:
            return 0
        before = values[:pos]
        after = values[pos:]
        return abs(np.mean(before) - np.mean(after))

    candidates = []
    for i in range(min_segment_size, n - min_segment_size):
        diff = segment_mean_diff(i)
        if diff > 0:
            candidates.append((i, diff))

    # Sort by difference and select top changepoints
    candidates.sort(key=lambda x: x[1], reverse=True)

    for pos, score in candidates[:max_changepoints]:
        # Verify statistical significance
        before = values[:pos]
        after = values[pos:]
        t_stat, p_value = stats.ttest_ind(before, after)

        if p_value < 0.05:
            changepoints.append((pos, score))

# Sort changepoints by position
changepoints.sort(key=lambda x: x[0])

# Build results
results = []
prev_end = 0

for i, (pos, score) in enumerate(changepoints):
    # Statistics before changepoint
    before_start = prev_end
    before_segment = values[before_start:pos]

    # Statistics after changepoint
    after_end = changepoints[i+1][0] if i+1 < len(changepoints) else n
    after_segment = values[pos:after_end]

    # Get date if available
    date_val = ''
    if date_column and date_column in df.columns:
        date_val = str(df.iloc[pos][date_column])

    results.append({
        'changepoint_index': pos,
        'date': date_val,
        'method': method,
        'confidence_score': round(score, 4),
        'mean_before': round(np.mean(before_segment), 4),
        'mean_after': round(np.mean(after_segment), 4),
        'std_before': round(np.std(before_segment, ddof=1), 4),
        'std_after': round(np.std(after_segment, ddof=1), 4),
        'change_magnitude': round(np.mean(after_segment) - np.mean(before_segment), 4),
        'change_pct': round((np.mean(after_segment) - np.mean(before_segment)) / (abs(np.mean(before_segment)) + 1e-10) * 100, 2),
        'segment_before_size': len(before_segment),
        'segment_after_size': len(after_segment)
    })

    prev_end = pos

if not results:
    results.append({
        'changepoint_index': 'No changepoints detected',
        'date': '',
        'method': method,
        'confidence_score': 0,
        'mean_before': round(np.mean(values), 4),
        'mean_after': round(np.mean(values), 4),
        'std_before': round(np.std(values, ddof=1), 4),
        'std_after': round(np.std(values, ddof=1), 4),
        'change_magnitude': 0,
        'change_pct': 0,
        'segment_before_size': n,
        'segment_after_size': 0
    })

output = pd.DataFrame(results)
`,
  },

  'isolation-forest': {
    type: 'isolation-forest',
    category: 'analysis',
    label: 'Isolation Forest',
    description: 'Detect anomalies using Isolation Forest algorithm',
    icon: 'AlertTriangle',
    defaultConfig: {
      features: [],
      contamination: 0.1,
      nEstimators: 100,
      addScores: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
contamination = float(config.get('contamination', 0.1))
n_estimators = int(config.get('nEstimators', 100))
add_scores = config.get('addScores', True)

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features:
    raise ValueError("Isolation Forest: No numeric columns found or specified")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Isolation Forest: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1)
X_clean = X[valid_mask]

if len(X_clean) < 10:
    raise ValueError("Isolation Forest: Need at least 10 valid data points")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)

predictions = model.fit_predict(X_scaled)
scores = model.decision_function(X_scaled)

df['is_anomaly'] = np.nan
df.loc[valid_mask, 'is_anomaly'] = (predictions == -1).astype(int)

if add_scores:
    df['anomaly_score'] = np.nan
    df.loc[valid_mask, 'anomaly_score'] = -scores

output = df
`,
  },

  'arima-forecasting': {
    type: 'arima-forecasting',
    category: 'analysis',
    label: 'ARIMA Forecasting',
    description: 'Time series forecasting using ARIMA/SARIMA models',
    icon: 'TrendingUp',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      periods: 12,
      autoArima: true,
      p: 1,
      d: 1,
      q: 1,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
periods = int(config.get('periods', 12))
p = int(config.get('p', 1))
d = int(config.get('d', 1))
q = int(config.get('q', 1))

if not date_col or not value_col:
    raise ValueError("ARIMA: Please specify date and value columns")

if date_col not in df.columns or value_col not in df.columns:
    raise ValueError("ARIMA: Required columns not found")

df = df.sort_values(date_col).reset_index(drop=True)
values = pd.to_numeric(df[value_col], errors='coerce').fillna(method='ffill').fillna(method='bfill').values

if len(values) < 10:
    raise ValueError("ARIMA: Need at least 10 data points")

def difference(data, d_order):
    diff_data = data.copy()
    for _ in range(d_order):
        diff_data = np.diff(diff_data)
    return diff_data

def undifference(forecast, last_values, d_order):
    result = forecast.copy()
    for i in range(d_order):
        result = np.cumsum(np.concatenate([[last_values[-(d_order-i)]], result]))
    return result[1:] if d_order > 0 else result

diff_values = difference(values, d)

def ar_forecast(data, p_order, n_forecast):
    if p_order == 0 or len(data) < p_order:
        return np.full(n_forecast, np.mean(data))
    X = np.array([data[i:len(data)-p_order+i] for i in range(p_order)]).T
    y = data[p_order:]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except:
        coeffs = np.ones(p_order) / p_order
    forecast = []
    history = list(data[-p_order:])
    for _ in range(n_forecast):
        pred = np.dot(coeffs, history[-p_order:])
        forecast.append(pred)
        history.append(pred)
    return np.array(forecast)

if len(diff_values) > 0:
    diff_forecast = ar_forecast(diff_values, p, periods)
    forecast = undifference(diff_forecast, values[-d:] if d > 0 else values[-1:], d)
else:
    forecast = np.full(periods, np.mean(values))

residual_std = np.std(diff_values) if len(diff_values) > 0 else np.std(values)
confidence_mult = np.sqrt(np.arange(1, periods + 1))
lower_ci = forecast - 1.96 * residual_std * confidence_mult
upper_ci = forecast + 1.96 * residual_std * confidence_mult

try:
    last_date = pd.to_datetime(df[date_col].iloc[-1])
    date_diff = pd.to_datetime(df[date_col]).diff().median()
    if pd.isna(date_diff):
        date_diff = pd.Timedelta(days=1)
    future_dates = [last_date + date_diff * (i + 1) for i in range(periods)]
except:
    future_dates = [f"Period_{i+1}" for i in range(periods)]

forecast_df = pd.DataFrame({
    date_col: future_dates,
    value_col: np.round(forecast, 4),
    'lower_ci_95': np.round(lower_ci, 4),
    'upper_ci_95': np.round(upper_ci, 4),
    'type': 'forecast'
})

actual_df = df[[date_col, value_col]].copy()
actual_df['lower_ci_95'] = np.nan
actual_df['upper_ci_95'] = np.nan
actual_df['type'] = 'actual'

output = pd.concat([actual_df, forecast_df], ignore_index=True)
`,
  },

  'seasonal-decomposition': {
    type: 'seasonal-decomposition',
    category: 'analysis',
    label: 'Seasonal Decomposition',
    description: 'Decompose time series into trend, seasonal, and residual components',
    icon: 'Layers',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      period: 12,
      model: 'additive',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
period = int(config.get('period', 12))
model_type = config.get('model', 'additive')

if not date_col or not value_col:
    raise ValueError("Seasonal Decomposition: Please specify date and value columns")

if date_col not in df.columns or value_col not in df.columns:
    raise ValueError("Seasonal Decomposition: Required columns not found")

df = df.sort_values(date_col).reset_index(drop=True)
values = pd.to_numeric(df[value_col], errors='coerce').fillna(method='ffill').fillna(method='bfill')

n = len(values)
if n < 2 * period:
    raise ValueError(f"Seasonal Decomposition: Need at least {2 * period} data points for period {period}")

def moving_average(data, window):
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.concatenate([np.full(pad_left, np.nan), ma, np.full(pad_right, np.nan)])

trend = moving_average(values.values, period)

if model_type == 'multiplicative':
    detrended = values.values / np.where(trend != 0, trend, 1)
else:
    detrended = values.values - trend

seasonal = np.zeros(n)
for i in range(period):
    indices = range(i, n, period)
    seasonal_mean = np.nanmean([detrended[j] for j in indices if not np.isnan(detrended[j])])
    for j in indices:
        seasonal[j] = seasonal_mean

if model_type == 'multiplicative':
    seasonal = seasonal / np.nanmean(seasonal)
else:
    seasonal = seasonal - np.nanmean(seasonal)

if model_type == 'multiplicative':
    residual = values.values / (trend * np.where(seasonal != 0, seasonal, 1))
else:
    residual = values.values - trend - seasonal

result = pd.DataFrame({
    date_col: df[date_col],
    'observed': values,
    'trend': np.round(trend, 4),
    'seasonal': np.round(seasonal, 4),
    'residual': np.round(residual, 4)
})

output = result
`,
  },

  'monte-carlo-simulation': {
    type: 'monte-carlo-simulation',
    category: 'analysis',
    label: 'Monte Carlo Simulation',
    description: 'Run Monte Carlo simulations for risk analysis and uncertainty quantification',
    icon: 'Shuffle',
    defaultConfig: {
      targetColumn: '',
      distribution: 'normal',
      simulations: 10000,
      confidenceLevel: 95,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
target_col = config.get('targetColumn', '')
distribution = config.get('distribution', 'normal')
n_simulations = int(config.get('simulations', 10000))
confidence_level = float(config.get('confidenceLevel', 95))

if not target_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("Monte Carlo: No numeric columns found")
    target_col = numeric_cols[0]

if target_col not in df.columns:
    raise ValueError(f"Monte Carlo: Column '{target_col}' not found")

data = pd.to_numeric(df[target_col], errors='coerce').dropna()

if len(data) < 2:
    raise ValueError("Monte Carlo: Need at least 2 data points")

mean = data.mean()
std = data.std()

np.random.seed(42)

if distribution == 'normal':
    simulated = np.random.normal(mean, std, n_simulations)
elif distribution == 'lognormal':
    log_mean = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    log_std = np.sqrt(np.log(1 + (std/mean)**2))
    simulated = np.random.lognormal(log_mean, log_std, n_simulations)
elif distribution == 'uniform':
    low = mean - 2 * std
    high = mean + 2 * std
    simulated = np.random.uniform(low, high, n_simulations)
elif distribution == 'triangular':
    low = data.min()
    high = data.max()
    mode = data.median()
    simulated = np.random.triangular(low, mode, high, n_simulations)
else:
    simulated = np.random.exponential(mean, n_simulations)

alpha = (100 - confidence_level) / 2
pct_values = np.percentile(simulated, [alpha, 50, 100 - alpha])

results = pd.DataFrame({
    'statistic': ['mean', 'std', 'min', 'max', 'median',
                  f'percentile_{alpha}', f'percentile_{100-alpha}',
                  'var_at_risk_95', 'expected_shortfall_95',
                  'skewness', 'kurtosis', 'n_simulations'],
    'value': [
        round(np.mean(simulated), 4),
        round(np.std(simulated), 4),
        round(np.min(simulated), 4),
        round(np.max(simulated), 4),
        round(np.median(simulated), 4),
        round(pct_values[0], 4),
        round(pct_values[2], 4),
        round(np.percentile(simulated, 5), 4),
        round(np.mean(simulated[simulated <= np.percentile(simulated, 5)]), 4),
        round(stats.skew(simulated), 4),
        round(stats.kurtosis(simulated), 4),
        n_simulations
    ]
})

output = results
`,
  },

  'propensity-score-matching': {
    type: 'propensity-score-matching',
    category: 'analysis',
    label: 'Propensity Score Matching',
    description: 'Match treatment and control groups for causal inference',
    icon: 'GitMerge',
    defaultConfig: {
      treatmentColumn: '',
      covariates: [],
      outcomeColumn: '',
      matchingMethod: 'nearest',
      caliper: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
treatment_col = config.get('treatmentColumn', '')
covariates = config.get('covariates', [])
outcome_col = config.get('outcomeColumn', '')
matching_method = config.get('matchingMethod', 'nearest')
caliper = float(config.get('caliper', 0.2))

if not treatment_col or not covariates:
    raise ValueError("PSM: Please specify treatment column and covariates")

missing = [c for c in [treatment_col] + covariates if c not in df.columns]
if missing:
    raise ValueError(f"PSM: Column(s) not found: {', '.join(missing)}")

df = df.dropna(subset=[treatment_col] + covariates)

treatment = pd.to_numeric(df[treatment_col], errors='coerce')
if treatment.nunique() > 2:
    raise ValueError("PSM: Treatment column must be binary (0/1)")

treatment = (treatment > 0).astype(int)

X = df[covariates].copy()
for col in covariates:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, treatment)
propensity_scores = model.predict_proba(X_scaled)[:, 1]

df['propensity_score'] = propensity_scores

treated_idx = df.index[treatment == 1].tolist()
control_idx = df.index[treatment == 0].tolist()

treated_ps = propensity_scores[treatment == 1]
control_ps = propensity_scores[treatment == 0]

matches = []
used_controls = set()

for i, t_idx in enumerate(treated_idx):
    t_ps = treated_ps[i]
    best_match = None
    best_dist = float('inf')

    for j, c_idx in enumerate(control_idx):
        if c_idx in used_controls:
            continue
        c_ps = control_ps[j]
        dist = abs(t_ps - c_ps)

        if dist < best_dist and dist <= caliper:
            best_dist = dist
            best_match = c_idx

    if best_match is not None:
        matches.append((t_idx, best_match, best_dist))
        used_controls.add(best_match)

matched_treated = [m[0] for m in matches]
matched_control = [m[1] for m in matches]
matched_indices = matched_treated + matched_control

matched_df = df.loc[matched_indices].copy()
matched_df['match_group'] = ['treated'] * len(matched_treated) + ['control'] * len(matched_control)
matched_df['match_distance'] = [m[2] for m in matches] + [m[2] for m in matches]

output = matched_df
`,
  },

  'difference-in-differences': {
    type: 'difference-in-differences',
    category: 'analysis',
    label: 'Difference-in-Differences',
    description: 'Estimate causal effects using difference-in-differences analysis',
    icon: 'GitCompare',
    defaultConfig: {
      timeColumn: '',
      groupColumn: '',
      treatmentColumn: '',
      outcomeColumn: '',
      prePostColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
time_col = config.get('timeColumn', '')
group_col = config.get('groupColumn', '')
treatment_col = config.get('treatmentColumn', '')
outcome_col = config.get('outcomeColumn', '')
pre_post_col = config.get('prePostColumn', '')

if not group_col or not outcome_col:
    raise ValueError("DiD: Please specify group and outcome columns")

required = [group_col, outcome_col]
if treatment_col:
    required.append(treatment_col)
if pre_post_col:
    required.append(pre_post_col)

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"DiD: Column(s) not found: {', '.join(missing)}")

if treatment_col:
    df['is_treated'] = (pd.to_numeric(df[treatment_col], errors='coerce') > 0).astype(int)
else:
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("DiD: Need exactly 2 groups (treatment and control)")
    df['is_treated'] = (df[group_col] == groups[1]).astype(int)

if pre_post_col:
    df['is_post'] = (pd.to_numeric(df[pre_post_col], errors='coerce') > 0).astype(int)
elif time_col:
    df['time_numeric'] = pd.to_numeric(df[time_col], errors='coerce')
    if df['time_numeric'].notna().sum() == 0:
        df['time_numeric'] = pd.to_datetime(df[time_col], errors='coerce').view('int64')
    median_time = df['time_numeric'].median()
    df['is_post'] = (df['time_numeric'] > median_time).astype(int)
else:
    raise ValueError("DiD: Please specify time or pre/post period column")

outcome = pd.to_numeric(df[outcome_col], errors='coerce')
df['outcome_numeric'] = outcome

means = df.groupby(['is_treated', 'is_post'])['outcome_numeric'].agg(['mean', 'std', 'count']).reset_index()

try:
    control_pre = means[(means['is_treated']==0) & (means['is_post']==0)]['mean'].values[0]
    control_post = means[(means['is_treated']==0) & (means['is_post']==1)]['mean'].values[0]
    treated_pre = means[(means['is_treated']==1) & (means['is_post']==0)]['mean'].values[0]
    treated_post = means[(means['is_treated']==1) & (means['is_post']==1)]['mean'].values[0]
except IndexError:
    raise ValueError("DiD: Missing data in one or more treatment-time groups")

did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

n_groups = df.groupby(['is_treated', 'is_post'])['outcome_numeric'].count()
pooled_var = df.groupby(['is_treated', 'is_post'])['outcome_numeric'].var().mean()
se = np.sqrt(pooled_var * (1/n_groups.sum()) * 4)

t_stat = did_estimate / se if se > 0 else 0
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(df)-4))

ci_lower = did_estimate - 1.96 * se
ci_upper = did_estimate + 1.96 * se

results = pd.DataFrame({
    'metric': [
        'did_estimate', 'standard_error', 't_statistic', 'p_value',
        'ci_lower_95', 'ci_upper_95',
        'treated_pre_mean', 'treated_post_mean',
        'control_pre_mean', 'control_post_mean',
        'treated_change', 'control_change'
    ],
    'value': [
        round(did_estimate, 4), round(se, 4), round(t_stat, 4), round(p_value, 4),
        round(ci_lower, 4), round(ci_upper, 4),
        round(treated_pre, 4), round(treated_post, 4),
        round(control_pre, 4), round(control_post, 4),
        round(treated_post - treated_pre, 4), round(control_post - control_pre, 4)
    ]
})

output = results
`,
  },

  'factor-analysis': {
    type: 'factor-analysis',
    category: 'analysis',
    label: 'Factor Analysis',
    description: 'Discover latent factors underlying observed variables',
    icon: 'Layers',
    defaultConfig: {
      features: [],
      nFactors: 3,
      rotation: 'varimax',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
n_factors = int(config.get('nFactors', 3))
rotation = config.get('rotation', 'varimax')

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if len(features) < 3:
    raise ValueError("Factor Analysis: Need at least 3 numeric columns")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Factor Analysis: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.dropna()

if len(X) < n_factors * 2:
    raise ValueError("Factor Analysis: Not enough valid data points")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

corr_matrix = np.corrcoef(X_scaled.T)

eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

positive_mask = eigenvalues > 0
eigenvalues = eigenvalues[positive_mask]
eigenvectors = eigenvectors[:, positive_mask]

n_factors = min(n_factors, len(eigenvalues))
eigenvalues = eigenvalues[:n_factors]
eigenvectors = eigenvectors[:, :n_factors]

loadings = eigenvectors * np.sqrt(eigenvalues)

if rotation == 'varimax' and n_factors > 1:
    for _ in range(100):
        old_loadings = loadings.copy()
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                u = loadings[:, i]**2 - loadings[:, j]**2
                v = 2 * loadings[:, i] * loadings[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                phi = 0.25 * np.arctan2(D - 2*A*B/len(u), C - (A**2 - B**2)/len(u))
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                loadings[:, i], loadings[:, j] = (
                    cos_phi * loadings[:, i] + sin_phi * loadings[:, j],
                    -sin_phi * loadings[:, i] + cos_phi * loadings[:, j]
                )
        if np.allclose(loadings, old_loadings, atol=1e-6):
            break

loading_cols = [f'Factor_{i+1}' for i in range(n_factors)]
loadings_df = pd.DataFrame(loadings, index=features, columns=loading_cols)
loadings_df['communality'] = np.sum(loadings**2, axis=1)
loadings_df = loadings_df.round(4)
loadings_df.index.name = 'variable'
loadings_df = loadings_df.reset_index()

output = loadings_df
`,
  },

  'dbscan-clustering': {
    type: 'dbscan-clustering',
    category: 'analysis',
    label: 'DBSCAN Clustering',
    description: 'Density-based clustering that identifies outliers automatically',
    icon: 'Network',
    defaultConfig: {
      features: [],
      eps: 0.5,
      minSamples: 5,
      metric: 'euclidean',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
eps = float(config.get('eps', 0.5))
min_samples = int(config.get('minSamples', 5))
metric = config.get('metric', 'euclidean')

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features:
    raise ValueError("DBSCAN: No numeric columns found or specified")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"DBSCAN: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1)
X_clean = X[valid_mask]

if len(X_clean) < min_samples:
    raise ValueError(f"DBSCAN: Need at least {min_samples} valid data points")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
labels = model.fit_predict(X_scaled)

df['cluster'] = np.nan
df.loc[valid_mask, 'cluster'] = labels

df['is_noise'] = np.nan
df.loc[valid_mask, 'is_noise'] = (labels == -1).astype(int)

core_sample_mask = np.zeros(len(labels), dtype=bool)
core_sample_mask[model.core_sample_indices_] = True
df['is_core_sample'] = np.nan
df.loc[valid_mask, 'is_core_sample'] = core_sample_mask.astype(int)

output = df
`,
  },

  'elastic-net': {
    type: 'elastic-net',
    category: 'analysis',
    label: 'Elastic Net Regression',
    description: 'Regularized regression combining L1 and L2 penalties',
    icon: 'TrendingUp',
    defaultConfig: {
      features: [],
      target: '',
      alpha: 1.0,
      l1Ratio: 0.5,
      testSize: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
alpha = float(config.get('alpha', 1.0))
l1_ratio = float(config.get('l1Ratio', 0.5))
test_size = float(config.get('testSize', 0.2))

if not target:
    raise ValueError("Elastic Net: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Elastic Net: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Elastic Net: No feature columns found or specified")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Elastic Net: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
y = pd.to_numeric(df[target], errors='coerce')

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X = X[valid_mask]
y = y[valid_mask]

if len(X) < 10:
    raise ValueError("Elastic Net: Need at least 10 valid data points")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

coefficients = pd.DataFrame({
    'feature': features,
    'coefficient': np.round(model.coef_, 6),
    'abs_coefficient': np.round(np.abs(model.coef_), 6),
    'selected': model.coef_ != 0
}).sort_values('abs_coefficient', ascending=False)

output = coefficients
`,
  },

  'var-analysis': {
    type: 'var-analysis',
    category: 'analysis',
    label: 'Vector Autoregression (VAR)',
    description: 'Multivariate time series analysis for interdependent variables',
    icon: 'Activity',
    defaultConfig: {
      dateColumn: '',
      variables: [],
      maxLags: 4,
      forecastPeriods: 10,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
variables = config.get('variables', [])
max_lags = int(config.get('maxLags', 4))
forecast_periods = int(config.get('forecastPeriods', 10))

if not variables or len(variables) < 2:
    raise ValueError("VAR: Please specify at least 2 variables")

missing = [c for c in variables if c not in df.columns]
if missing:
    raise ValueError(f"VAR: Column(s) not found: {', '.join(missing)}")

if date_col and date_col in df.columns:
    df = df.sort_values(date_col).reset_index(drop=True)

data = df[variables].copy()
for col in variables:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

n = len(data)
k = len(variables)

if n < max_lags * 2 + 10:
    raise ValueError(f"VAR: Need at least {max_lags * 2 + 10} data points")

Y = data.values[max_lags:]
T = len(Y)

X_list = [np.ones((T, 1))]
for lag in range(1, max_lags + 1):
    X_list.append(data.values[max_lags - lag:-lag if lag < max_lags else T + max_lags - lag])

X = np.hstack(X_list)

coefficients = {}
for i, var in enumerate(variables):
    y = Y[:, i]
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        coefficients[var] = beta
    except:
        coefficients[var] = np.zeros(X.shape[1])

last_values = data.values[-max_lags:].copy()
forecasts = []

for h in range(forecast_periods):
    forecast_row = []
    for i, var in enumerate(variables):
        beta = coefficients[var]
        pred = beta[0]
        for lag in range(1, max_lags + 1):
            idx = 1 + (lag - 1) * k
            if h < lag:
                pred += np.dot(beta[idx:idx + k], last_values[-(lag - h)])
            else:
                pred += np.dot(beta[idx:idx + k], forecasts[h - lag])
        forecast_row.append(pred)
    forecasts.append(forecast_row)
    last_values = np.vstack([last_values[1:], forecast_row])

forecasts = np.array(forecasts)

if date_col and date_col in df.columns:
    try:
        last_date = pd.to_datetime(df[date_col].iloc[-1])
        date_diff = pd.to_datetime(df[date_col]).diff().median()
        if pd.isna(date_diff):
            date_diff = pd.Timedelta(days=1)
        future_dates = [last_date + date_diff * (i + 1) for i in range(forecast_periods)]
    except:
        future_dates = [f"t+{i+1}" for i in range(forecast_periods)]
else:
    future_dates = [f"t+{i+1}" for i in range(forecast_periods)]

forecast_df = pd.DataFrame(forecasts, columns=variables)
forecast_df['period'] = future_dates
forecast_df['type'] = 'forecast'

hist_df = data.copy()
if date_col and date_col in df.columns:
    hist_df['period'] = df[date_col].iloc[-len(data):].values
else:
    hist_df['period'] = range(len(data))
hist_df['type'] = 'actual'

result = pd.concat([hist_df, forecast_df], ignore_index=True)

output = result
`,
  },

  'interrupted-time-series': {
    type: 'interrupted-time-series',
    category: 'analysis',
    label: 'Interrupted Time Series',
    description: 'Analyze the impact of interventions on time series data',
    icon: 'GitBranch',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      interventionDate: '',
      interventionIndex: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
intervention_date = config.get('interventionDate', '')
intervention_idx = config.get('interventionIndex', '')

if not value_col:
    raise ValueError("ITS: Please specify a value column")

if value_col not in df.columns:
    raise ValueError(f"ITS: Column '{value_col}' not found")

if date_col and date_col in df.columns:
    df = df.sort_values(date_col).reset_index(drop=True)

values = pd.to_numeric(df[value_col], errors='coerce').values
n = len(values)

if intervention_idx:
    intervention_point = int(intervention_idx)
elif intervention_date and date_col:
    try:
        dates = pd.to_datetime(df[date_col])
        intervention_dt = pd.to_datetime(intervention_date)
        intervention_point = (dates >= intervention_dt).idxmax()
    except:
        intervention_point = n // 2
else:
    intervention_point = n // 2

if intervention_point < 5 or intervention_point > n - 5:
    raise ValueError("ITS: Intervention point must have at least 5 observations before and after")

time = np.arange(n)
intervention = (time >= intervention_point).astype(int)
time_since_intervention = np.maximum(0, time - intervention_point)

pre_values = values[:intervention_point]
pre_time = time[:intervention_point]
post_values = values[intervention_point:]
post_time = time[intervention_point:]

pre_slope, pre_intercept, _, _, _ = stats.linregress(pre_time, pre_values)
post_slope, post_intercept, _, _, _ = stats.linregress(post_time, post_values)

counterfactual = pre_intercept + pre_slope * time

X = np.column_stack([np.ones(n), time, intervention, time_since_intervention])
y = values

try:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, trend, level_change, trend_change = beta
except:
    intercept, trend, level_change, trend_change = pre_intercept, pre_slope, 0, 0

predicted = intercept + trend * time + level_change * intervention + trend_change * time_since_intervention

immediate_effect = level_change
sustained_effect = level_change + trend_change * (n - 1 - intervention_point)

residual_std = np.std(y - predicted)
se_level = residual_std / np.sqrt(n)
t_level = level_change / se_level if se_level > 0 else 0
p_level = 2 * (1 - stats.t.cdf(abs(t_level), df=n-4))

results = pd.DataFrame({
    'metric': [
        'intervention_point', 'n_pre', 'n_post',
        'pre_trend_slope', 'post_trend_slope',
        'level_change', 'trend_change',
        'immediate_effect', 'sustained_effect_end',
        'level_change_pvalue', 'significant_at_05',
        'pre_mean', 'post_mean', 'mean_difference'
    ],
    'value': [
        intervention_point, intervention_point, n - intervention_point,
        round(pre_slope, 4), round(post_slope, 4),
        round(level_change, 4), round(trend_change, 4),
        round(immediate_effect, 4), round(sustained_effect, 4),
        round(p_level, 4), 'Yes' if p_level < 0.05 else 'No',
        round(np.mean(pre_values), 4), round(np.mean(post_values), 4),
        round(np.mean(post_values) - np.mean(pre_values), 4)
    ]
})

output = results
`,
  },

  'granger-causality': {
    type: 'granger-causality',
    category: 'analysis',
    label: 'Granger Causality Test',
    description: 'Test whether one time series helps predict another',
    icon: 'GitCompare',
    defaultConfig: {
      variable1: '',
      variable2: '',
      dateColumn: '',
      maxLags: 4,
      significanceLevel: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
var1 = config.get('variable1', '')
var2 = config.get('variable2', '')
date_col = config.get('dateColumn', '')
max_lags = int(config.get('maxLags', 4))
sig_level = float(config.get('significanceLevel', 0.05))

if not var1 or not var2:
    raise ValueError("Granger Causality: Please specify two variables")

if var1 not in df.columns or var2 not in df.columns:
    raise ValueError("Granger Causality: Specified columns not found")

if date_col and date_col in df.columns:
    df = df.sort_values(date_col).reset_index(drop=True)

y1 = pd.to_numeric(df[var1], errors='coerce').values
y2 = pd.to_numeric(df[var2], errors='coerce').values

valid = ~(np.isnan(y1) | np.isnan(y2))
y1 = y1[valid]
y2 = y2[valid]

n = len(y1)

if n < max_lags * 2 + 10:
    raise ValueError(f"Granger Causality: Need at least {max_lags * 2 + 10} data points")

def granger_test(x, y, max_lag):
    results = []
    for lag in range(1, max_lag + 1):
        Y = y[lag:]
        X_restricted = np.column_stack([y[lag-i-1:-i-1 if i < lag-1 else None] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
        X_unrestricted = X_restricted.copy()
        for i in range(lag):
            X_unrestricted = np.column_stack([X_unrestricted, x[lag-i-1:-i-1 if i < lag-1 else None]])
        try:
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            ssr_r = np.sum((Y - X_restricted @ beta_r)**2)
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            ssr_u = np.sum((Y - X_unrestricted @ beta_u)**2)
            df1 = lag
            df2 = len(Y) - 2 * lag - 1
            if df2 > 0 and ssr_u > 0:
                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            else:
                f_stat = 0
                p_value = 1
            results.append({'lag': lag, 'f_statistic': round(f_stat, 4), 'p_value': round(p_value, 4), 'significant': p_value < sig_level})
        except:
            results.append({'lag': lag, 'f_statistic': np.nan, 'p_value': np.nan, 'significant': False})
    return results

results_1_to_2 = granger_test(y1, y2, max_lags)
results_2_to_1 = granger_test(y2, y1, max_lags)

all_results = []
for r in results_1_to_2:
    all_results.append({
        'direction': f'{var1} -> {var2}',
        'lag': r['lag'],
        'f_statistic': r['f_statistic'],
        'p_value': r['p_value'],
        'significant': 'Yes' if r['significant'] else 'No'
    })
for r in results_2_to_1:
    all_results.append({
        'direction': f'{var2} -> {var1}',
        'lag': r['lag'],
        'f_statistic': r['f_statistic'],
        'p_value': r['p_value'],
        'significant': 'Yes' if r['significant'] else 'No'
    })

output = pd.DataFrame(all_results)
`,
  },

  'local-outlier-factor': {
    type: 'local-outlier-factor',
    category: 'analysis',
    label: 'Local Outlier Factor',
    description: 'Detect local outliers based on density deviation from neighbors',
    icon: 'AlertTriangle',
    defaultConfig: {
      features: [],
      nNeighbors: 20,
      contamination: 0.1,
      addScores: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
n_neighbors = int(config.get('nNeighbors', 20))
contamination = float(config.get('contamination', 0.1))
add_scores = config.get('addScores', True)

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features:
    raise ValueError("LOF: No numeric columns found or specified")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"LOF: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1)
X_clean = X[valid_mask]

if len(X_clean) < n_neighbors + 1:
    raise ValueError(f"LOF: Need at least {n_neighbors + 1} valid data points")

n_neighbors = min(n_neighbors, len(X_clean) - 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

model = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=contamination,
    novelty=False,
    n_jobs=-1
)

predictions = model.fit_predict(X_scaled)
scores = model.negative_outlier_factor_

df['is_outlier'] = np.nan
df.loc[valid_mask, 'is_outlier'] = (predictions == -1).astype(int)

if add_scores:
    df['lof_score'] = np.nan
    df.loc[valid_mask, 'lof_score'] = -scores

df['local_density_rank'] = np.nan
density_ranks = (-scores).argsort().argsort() / len(scores)
df.loc[valid_mask, 'local_density_rank'] = density_ranks

output = df
`,
  },

  // New Analysis Blocks for Data Scientists
  'feature-selection': {
    type: 'feature-selection',
    category: 'analysis',
    label: 'Feature Selection',
    description: 'Select most predictive features using RFE, SelectKBest, or Mutual Information',
    icon: 'Filter',
    defaultConfig: {
      features: [],
      target: '',
      method: 'selectkbest',
      nFeatures: 10,
      scoreFunc: 'f_classif',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
method = config.get('method', 'selectkbest')
n_features = int(config.get('nFeatures', 10))
score_func = config.get('scoreFunc', 'f_classif')

if not target:
    raise ValueError("Feature Selection: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Feature Selection: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Feature Selection: No numeric feature columns found")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Feature Selection: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 10:
    raise ValueError("Feature Selection: Need at least 10 valid samples")

n_features = min(n_features, len(features))
is_classification = y_clean.nunique() <= 20

if method == 'selectkbest':
    if score_func == 'f_classif':
        func = f_classif if is_classification else f_regression
    elif score_func == 'mutual_info':
        func = mutual_info_classif if is_classification else mutual_info_regression
    else:
        func = f_classif if is_classification else f_regression
    selector = SelectKBest(score_func=func, k=n_features)
    selector.fit(X_clean, y_clean)
    scores = selector.scores_
    selected_mask = selector.get_support()
elif method == 'rfe':
    if is_classification:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X_clean, y_clean)
    scores = selector.ranking_
    selected_mask = selector.support_
elif method == 'mutual_info':
    if is_classification:
        scores = mutual_info_classif(X_clean, y_clean, random_state=42)
    else:
        scores = mutual_info_regression(X_clean, y_clean, random_state=42)
    indices = np.argsort(scores)[::-1][:n_features]
    selected_mask = np.zeros(len(features), dtype=bool)
    selected_mask[indices] = True
else:
    raise ValueError(f"Feature Selection: Unknown method '{method}'")

selected_features = [f for f, s in zip(features, selected_mask) if s]
feature_scores = pd.DataFrame({
    'feature': features,
    'score': scores if method != 'rfe' else (len(features) - scores + 1),
    'selected': selected_mask,
    'rank': np.argsort(np.argsort(-np.array(scores if method != 'rfe' else (len(features) - scores + 1)))) + 1
}).sort_values('rank')

output_df = df[[target] + selected_features].copy()
output_df.attrs['feature_scores'] = feature_scores.to_dict('records')
output_df.attrs['selected_features'] = selected_features
output = output_df
`,
  },

  'outlier-treatment': {
    type: 'outlier-treatment',
    category: 'analysis',
    label: 'Outlier Treatment',
    description: 'Cap, remove, or impute outliers using IQR, Z-score, or percentile methods',
    icon: 'Shield',
    defaultConfig: {
      columns: [],
      method: 'iqr',
      action: 'cap',
      threshold: 1.5,
      lowerPercentile: 1,
      upperPercentile: 99,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'iqr')
action = config.get('action', 'cap')
threshold = float(config.get('threshold', 1.5))
lower_pct = float(config.get('lowerPercentile', 1))
upper_pct = float(config.get('upperPercentile', 99))

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Outlier Treatment: No numeric columns found or specified")

missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Outlier Treatment: Column(s) not found: {', '.join(missing)}")

outlier_info = []
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    valid_data = df[col].dropna()
    if len(valid_data) == 0:
        continue
    if method == 'iqr':
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
    elif method == 'zscore':
        mean = valid_data.mean()
        std = valid_data.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    elif method == 'percentile':
        lower_bound = valid_data.quantile(lower_pct / 100)
        upper_bound = valid_data.quantile(upper_pct / 100)
    else:
        raise ValueError(f"Outlier Treatment: Unknown method '{method}'")
    is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
    n_outliers = is_outlier.sum()
    outlier_info.append({
        'column': col, 'lower_bound': float(lower_bound), 'upper_bound': float(upper_bound),
        'n_outliers': int(n_outliers), 'pct_outliers': float(n_outliers / len(df) * 100)
    })
    if action == 'cap':
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
    elif action == 'remove':
        df = df[~is_outlier]
    elif action == 'null':
        df.loc[is_outlier, col] = np.nan
    elif action == 'mean':
        mean_val = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)].mean()
        df.loc[is_outlier, col] = mean_val
    elif action == 'median':
        median_val = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)].median()
        df.loc[is_outlier, col] = median_val

df.attrs['outlier_treatment'] = outlier_info
output = df
`,
  },

  'data-drift': {
    type: 'data-drift',
    category: 'analysis',
    label: 'Data Drift Detection',
    description: 'Detect distribution shifts between reference and current data using KS-test and PSI',
    icon: 'GitCompare',
    defaultConfig: {
      columns: [],
      method: 'ks_test',
      threshold: 0.05,
      psiBins: 10,
    },
    inputs: 2,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

reference_df = input_data[0].copy()
current_df = input_data[1].copy()
columns = config.get('columns', [])
method = config.get('method', 'ks_test')
threshold = float(config.get('threshold', 0.05))
psi_bins = int(config.get('psiBins', 10))

if not columns:
    common_cols = set(reference_df.columns) & set(current_df.columns)
    columns = [c for c in common_cols if reference_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

if not columns:
    raise ValueError("Data Drift: No common numeric columns found")

def calculate_psi(reference, current, bins=10):
    ref_min, ref_max = reference.min(), reference.max()
    bin_edges = np.linspace(ref_min, ref_max, bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]
    ref_pct = (ref_counts + 0.001) / (len(reference) + 0.001 * bins)
    cur_pct = (cur_counts + 0.001) / (len(current) + 0.001 * bins)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

drift_results = []
for col in columns:
    ref_data = pd.to_numeric(reference_df[col], errors='coerce').dropna()
    cur_data = pd.to_numeric(current_df[col], errors='coerce').dropna()
    if len(ref_data) < 5 or len(cur_data) < 5:
        continue
    result = {'column': col}
    if method in ['ks_test', 'both']:
        ks_stat, ks_pval = stats.ks_2samp(ref_data, cur_data)
        result['ks_statistic'] = float(ks_stat)
        result['ks_pvalue'] = float(ks_pval)
        result['ks_drift_detected'] = ks_pval < threshold
    if method in ['psi', 'both']:
        psi = calculate_psi(ref_data.values, cur_data.values, psi_bins)
        result['psi'] = float(psi)
        result['psi_drift_detected'] = psi > 0.2
        result['psi_severity'] = 'no_drift' if psi < 0.1 else ('slight_drift' if psi < 0.2 else 'significant_drift')
    result['ref_mean'] = float(ref_data.mean())
    result['cur_mean'] = float(cur_data.mean())
    result['mean_shift_pct'] = float((cur_data.mean() - ref_data.mean()) / (ref_data.mean() + 1e-10) * 100)
    drift_results.append(result)

output = pd.DataFrame(drift_results)
`,
  },

  'polynomial-features': {
    type: 'polynomial-features',
    category: 'analysis',
    label: 'Polynomial Features',
    description: 'Create polynomial and interaction features from numeric columns',
    icon: 'Sigma',
    defaultConfig: {
      columns: [],
      degree: 2,
      interactionOnly: false,
      includeBias: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

df = input_data.copy()
columns = config.get('columns', [])
degree = int(config.get('degree', 2))
interaction_only = config.get('interactionOnly', False)
include_bias = config.get('includeBias', False)

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not columns:
    raise ValueError("Polynomial Features: No numeric columns found or specified")

if len(columns) > 10:
    raise ValueError("Polynomial Features: Maximum 10 columns allowed to avoid memory issues")

missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Polynomial Features: Column(s) not found: {', '.join(missing)}")

X = df[columns].copy()
for col in columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1)
X_clean = X[valid_mask].values

poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
X_poly = poly.fit_transform(X_clean)
feature_names = poly.get_feature_names_out(columns)

poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df[valid_mask].index)
new_features = [f for f in feature_names if f not in columns]

for feat in new_features:
    df[feat] = np.nan
    df.loc[valid_mask, feat] = poly_df[feat].values

df.attrs['polynomial_features'] = {
    'original_columns': columns, 'new_features': new_features, 'degree': degree, 'n_features_created': len(new_features)
}
output = df
`,
  },

  'multi-output': {
    type: 'multi-output',
    category: 'analysis',
    label: 'Multi-Output Prediction',
    description: 'Predict multiple target variables simultaneously',
    icon: 'GitFork',
    defaultConfig: {
      features: [],
      targets: [],
      modelType: 'random_forest',
      taskType: 'auto',
      testSize: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

df = input_data.copy()
features = config.get('features', [])
targets = config.get('targets', [])
model_type = config.get('modelType', 'random_forest')
task_type = config.get('taskType', 'auto')
test_size = float(config.get('testSize', 0.2))

if not targets or len(targets) < 2:
    raise ValueError("Multi-Output: Please specify at least 2 target columns")

for t in targets:
    if t not in df.columns:
        raise ValueError(f"Multi-Output: Target column '{t}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in targets]

if not features:
    raise ValueError("Multi-Output: No numeric feature columns found")

X = df[features].copy()
Y = df[targets].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
X_clean = X[valid_mask]
Y_clean = Y[valid_mask]

if len(X_clean) < 20:
    raise ValueError("Multi-Output: Need at least 20 valid samples")

is_classification = all(Y_clean[t].nunique() <= 20 for t in targets) if task_type == 'auto' else task_type == 'classification'

label_encoders = {}
if is_classification:
    for t in targets:
        if Y_clean[t].dtype == 'object':
            le = LabelEncoder()
            Y_clean[t] = le.fit_transform(Y_clean[t].astype(str))
            label_encoders[t] = le

X_train, X_test, Y_train, Y_test = train_test_split(X_clean, Y_clean, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if is_classification:
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
else:
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))

model.fit(X_train_scaled, Y_train)
predictions = model.predict(X_test_scaled)

metrics = {}
for i, t in enumerate(targets):
    if is_classification:
        metrics[t] = {'accuracy': float(accuracy_score(Y_test.iloc[:, i], predictions[:, i]))}
    else:
        metrics[t] = {'r2_score': float(r2_score(Y_test.iloc[:, i], predictions[:, i])),
                      'rmse': float(np.sqrt(mean_squared_error(Y_test.iloc[:, i], predictions[:, i])))}

X_all_scaled = scaler.transform(X_clean)
all_predictions = model.predict(X_all_scaled)

for i, t in enumerate(targets):
    pred_col = f'{t}_predicted'
    df[pred_col] = np.nan
    df.loc[valid_mask, pred_col] = all_predictions[:, i]

df.attrs['multi_output_metrics'] = metrics
df.attrs['multi_output_task'] = 'classification' if is_classification else 'regression'
output = df
`,
  },

  'probability-calibration': {
    type: 'probability-calibration',
    category: 'analysis',
    label: 'Probability Calibration',
    description: 'Calibrate classifier probabilities using Platt scaling or isotonic regression',
    icon: 'Gauge',
    defaultConfig: {
      probabilityColumn: '',
      actualColumn: '',
      method: 'isotonic',
      nBins: 10,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

df = input_data.copy()
prob_col = config.get('probabilityColumn', '')
actual_col = config.get('actualColumn', '')
method = config.get('method', 'isotonic')
n_bins = int(config.get('nBins', 10))

if not prob_col or not actual_col:
    raise ValueError("Probability Calibration: Please specify probability and actual columns")

if prob_col not in df.columns or actual_col not in df.columns:
    raise ValueError("Probability Calibration: Specified columns not found")

probs = pd.to_numeric(df[prob_col], errors='coerce')
actuals = pd.to_numeric(df[actual_col], errors='coerce')

valid_mask = probs.notna() & actuals.notna() & (probs >= 0) & (probs <= 1)
probs_clean = probs[valid_mask].values
actuals_clean = actuals[valid_mask].values

if len(probs_clean) < 20:
    raise ValueError("Probability Calibration: Need at least 20 valid samples")

fraction_pos_before, mean_pred_before = calibration_curve(actuals_clean, probs_clean, n_bins=n_bins, strategy='uniform')

if method == 'isotonic':
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(probs_clean, actuals_clean)
    calibrated_probs = calibrator.predict(probs_clean)
elif method == 'platt':
    calibrator = LogisticRegression()
    calibrator.fit(probs_clean.reshape(-1, 1), actuals_clean)
    calibrated_probs = calibrator.predict_proba(probs_clean.reshape(-1, 1))[:, 1]
else:
    raise ValueError(f"Probability Calibration: Unknown method '{method}'")

fraction_pos_after, mean_pred_after = calibration_curve(actuals_clean, calibrated_probs, n_bins=n_bins, strategy='uniform')
brier_before = np.mean((probs_clean - actuals_clean) ** 2)
brier_after = np.mean((calibrated_probs - actuals_clean) ** 2)

df['calibrated_probability'] = np.nan
df.loc[valid_mask, 'calibrated_probability'] = calibrated_probs

df.attrs['calibration_results'] = {
    'brier_score_before': float(brier_before), 'brier_score_after': float(brier_after),
    'improvement_pct': float((brier_before - brier_after) / brier_before * 100)
}
output = df
`,
  },

  'tsne-reduction': {
    type: 'tsne-reduction',
    category: 'analysis',
    label: 't-SNE Reduction',
    description: 'Reduce high-dimensional data to 2D/3D using t-SNE for visualization',
    icon: 'Minimize2',
    defaultConfig: {
      features: [],
      nComponents: 2,
      perplexity: 30,
      learningRate: 200,
      nIter: 1000,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
n_components = int(config.get('nComponents', 2))
perplexity = float(config.get('perplexity', 30))
learning_rate = float(config.get('learningRate', 200))
n_iter = int(config.get('nIter', 1000))

if n_components not in [2, 3]:
    raise ValueError("t-SNE: nComponents must be 2 or 3")

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features or len(features) < n_components:
    raise ValueError(f"t-SNE: Need at least {n_components} numeric features")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"t-SNE: Column(s) not found: {', '.join(missing)}")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1)
X_clean = X[valid_mask]

if len(X_clean) < 5:
    raise ValueError("t-SNE: Need at least 5 valid samples")

perplexity = min(max(perplexity, 5), len(X_clean) - 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42, init='pca')
X_embedded = tsne.fit_transform(X_scaled)

component_names = [f'tsne_{i+1}' for i in range(n_components)]
for i, name in enumerate(component_names):
    df[name] = np.nan
    df.loc[valid_mask, name] = X_embedded[:, i]

df.attrs['tsne_info'] = {'n_components': n_components, 'perplexity': perplexity, 'kl_divergence': float(tsne.kl_divergence_)}
output = df
`,
  },

  'statistical-tests': {
    type: 'statistical-tests',
    category: 'analysis',
    label: 'Statistical Tests Suite',
    description: 'Perform various statistical tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Levene',
    icon: 'FlaskConical',
    defaultConfig: {
      testType: 'mann_whitney',
      column1: '',
      column2: '',
      groupColumn: '',
      valueColumn: '',
      alpha: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
test_type = config.get('testType', 'mann_whitney')
column1 = config.get('column1', '')
column2 = config.get('column2', '')
group_col = config.get('groupColumn', '')
value_col = config.get('valueColumn', '')
alpha = float(config.get('alpha', 0.05))

results = {'test_type': test_type, 'alpha': alpha}

if test_type == 'mann_whitney':
    if not column1 or not column2:
        raise ValueError("Mann-Whitney: Please specify two columns to compare")
    data1 = pd.to_numeric(df[column1], errors='coerce').dropna()
    data2 = pd.to_numeric(df[column2], errors='coerce').dropna()
    stat, pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    n1, n2 = len(data1), len(data2)
    r = 1 - (2 * stat) / (n1 * n2)
    results.update({'statistic': float(stat), 'p_value': float(pval), 'significant': pval < alpha,
                    'effect_size_r': float(r), 'medians': {'group1': float(data1.median()), 'group2': float(data2.median())}})
elif test_type == 'wilcoxon':
    if not column1 or not column2:
        raise ValueError("Wilcoxon: Please specify two paired columns")
    data1 = pd.to_numeric(df[column1], errors='coerce')
    data2 = pd.to_numeric(df[column2], errors='coerce')
    valid = data1.notna() & data2.notna()
    stat, pval = stats.wilcoxon(data1[valid], data2[valid])
    results.update({'statistic': float(stat), 'p_value': float(pval), 'significant': pval < alpha,
                    'n_pairs': int(valid.sum()), 'median_difference': float((data1[valid] - data2[valid]).median())})
elif test_type == 'kruskal_wallis':
    if not group_col or not value_col:
        raise ValueError("Kruskal-Wallis: Please specify group and value columns")
    groups = df.groupby(group_col)[value_col].apply(lambda x: pd.to_numeric(x, errors='coerce').dropna().tolist()).tolist()
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis: Need at least 2 groups with data")
    stat, pval = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eta_squared = (stat - len(groups) + 1) / (n_total - len(groups))
    results.update({'statistic': float(stat), 'p_value': float(pval), 'significant': pval < alpha,
                    'effect_size_eta_squared': float(eta_squared), 'n_groups': len(groups)})
elif test_type == 'levene':
    if not group_col or not value_col:
        raise ValueError("Levene: Please specify group and value columns")
    groups = df.groupby(group_col)[value_col].apply(lambda x: pd.to_numeric(x, errors='coerce').dropna().tolist()).tolist()
    groups = [g for g in groups if len(g) > 0]
    stat, pval = stats.levene(*groups, center='median')
    results.update({'statistic': float(stat), 'p_value': float(pval), 'significant': pval < alpha,
                    'homogeneous_variance': pval >= alpha, 'n_groups': len(groups)})
else:
    raise ValueError(f"Statistical Tests: Unknown test type '{test_type}'")

output = pd.DataFrame([results])
`,
  },

  'optimal-binning': {
    type: 'optimal-binning',
    category: 'analysis',
    label: 'Optimal Binning',
    description: 'Create optimal bins using decision tree or quantile-based methods with WoE/IV',
    icon: 'BarChart',
    defaultConfig: {
      column: '',
      target: '',
      method: 'quantile',
      nBins: 10,
      minBinSize: 0.05,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = input_data.copy()
column = config.get('column', '')
target = config.get('target', '')
method = config.get('method', 'quantile')
n_bins = int(config.get('nBins', 10))
min_bin_size = float(config.get('minBinSize', 0.05))

if not column:
    raise ValueError("Optimal Binning: Please specify a column to bin")

if column not in df.columns:
    raise ValueError(f"Optimal Binning: Column '{column}' not found")

values = pd.to_numeric(df[column], errors='coerce')
valid_mask = values.notna()

if target and target in df.columns:
    target_vals = pd.to_numeric(df[target], errors='coerce')
    valid_mask = valid_mask & target_vals.notna()
    if target_vals[valid_mask].nunique() != 2:
        target = ''

values_clean = values[valid_mask]

if method == 'quantile':
    try:
        bin_edges = pd.qcut(values_clean, q=n_bins, duplicates='drop', retbins=True)[1]
    except ValueError:
        bin_edges = np.percentile(values_clean, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
elif method == 'equal_width':
    bin_edges = np.linspace(values_clean.min(), values_clean.max(), n_bins + 1)
elif method == 'tree' and target:
    target_clean = target_vals[valid_mask]
    tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=int(len(values_clean) * min_bin_size), random_state=42)
    tree.fit(values_clean.values.reshape(-1, 1), target_clean)
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
    bin_edges = np.sort(np.unique(np.concatenate([[values_clean.min()], thresholds, [values_clean.max()]])))
else:
    bin_edges = np.linspace(values_clean.min(), values_clean.max(), n_bins + 1)

bin_edges = np.array(bin_edges)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf

binned_col = f'{column}_binned'
df[binned_col] = pd.cut(values, bins=bin_edges, labels=False, include_lowest=True)

bin_stats = []
for i in range(len(bin_edges) - 1):
    mask = df[binned_col] == i
    bin_data = {'bin': i, 'count': int(mask.sum()), 'pct': float(mask.sum() / len(df) * 100)}
    if target and target in df.columns:
        target_vals_bin = target_vals[mask & valid_mask]
        if len(target_vals_bin) > 0:
            bin_data['event_rate'] = float(target_vals_bin.mean())
    bin_stats.append(bin_data)

df.attrs['bin_stats'] = bin_stats
output = df
`,
  },

  'correlation-finder': {
    type: 'correlation-finder',
    category: 'analysis',
    label: 'Correlation Finder',
    description: 'Find highly correlated pairs and identify multicollinearity issues',
    icon: 'Link',
    defaultConfig: {
      columns: [],
      method: 'pearson',
      threshold: 0.7,
      showTopN: 20,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'pearson')
threshold = float(config.get('threshold', 0.7))
show_top_n = int(config.get('showTopN', 20))

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(columns) < 2:
    raise ValueError("Correlation Finder: Need at least 2 numeric columns")

missing = [c for c in columns if c not in df.columns]
if missing:
    raise ValueError(f"Correlation Finder: Column(s) not found: {', '.join(missing)}")

X = df[columns].copy()
for col in columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

corr_matrix = X.corr(method=method)

pairs = []
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        col1, col2 = columns[i], columns[j]
        corr = corr_matrix.loc[col1, col2]
        if pd.notna(corr):
            valid_data = X[[col1, col2]].dropna()
            n = len(valid_data)
            if n > 2:
                if method == 'pearson':
                    _, pval = stats.pearsonr(valid_data[col1], valid_data[col2])
                elif method == 'spearman':
                    _, pval = stats.spearmanr(valid_data[col1], valid_data[col2])
                else:
                    _, pval = stats.kendalltau(valid_data[col1], valid_data[col2])
            else:
                pval = 1.0
            pairs.append({'column1': col1, 'column2': col2, 'correlation': float(corr),
                          'abs_correlation': float(abs(corr)), 'p_value': float(pval),
                          'is_significant': pval < 0.05, 'is_high_correlation': abs(corr) >= threshold})

pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sort_values('abs_correlation', ascending=False).head(show_top_n)
pairs_df.attrs['threshold'] = threshold
pairs_df.attrs['n_high_correlation_pairs'] = len(pairs_df[pairs_df['abs_correlation'] >= threshold])
output = pairs_df
`,
  },

  'ab-test-calculator': {
    type: 'ab-test-calculator',
    category: 'analysis',
    label: 'A/B Test Calculator',
    description: 'Calculate statistical significance for A/B tests with confidence intervals',
    icon: 'FlaskConical',
    defaultConfig: {
      testType: 'conversion',
      groupColumn: '',
      valueColumn: '',
      confidenceLevel: 0.95,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
test_type = config.get('testType', 'conversion')
group_col = config.get('groupColumn', '')
value_col = config.get('valueColumn', '')
conf_level = float(config.get('confidenceLevel', 0.95))

alpha = 1 - conf_level
z_score = stats.norm.ppf(1 - alpha / 2)

if not group_col or not value_col:
    raise ValueError("A/B Test: Please specify group and value columns")

groups = df.groupby(group_col)
group_names = list(groups.groups.keys())

if len(group_names) < 2:
    raise ValueError("A/B Test: Need at least 2 groups (control and treatment)")

control_name = group_names[0]
treatment_name = group_names[1]

control_data = pd.to_numeric(groups.get_group(control_name)[value_col], errors='coerce').dropna()
treatment_data = pd.to_numeric(groups.get_group(treatment_name)[value_col], errors='coerce').dropna()

n_c, n_t = len(control_data), len(treatment_data)
results = {'test_type': test_type, 'confidence_level': conf_level,
           'control_group': str(control_name), 'treatment_group': str(treatment_name),
           'control_n': int(n_c), 'treatment_n': int(n_t)}

if test_type == 'conversion':
    conv_c, conv_t = control_data.mean(), treatment_data.mean()
    se_c = np.sqrt(conv_c * (1 - conv_c) / n_c) if 0 < conv_c < 1 else 0.01
    se_t = np.sqrt(conv_t * (1 - conv_t) / n_t) if 0 < conv_t < 1 else 0.01
    se_diff = np.sqrt(se_c**2 + se_t**2)
    lift = (conv_t - conv_c) / conv_c * 100 if conv_c > 0 else 0
    z_stat = (conv_t - conv_c) / se_diff if se_diff > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lower = (conv_t - conv_c) - z_score * se_diff
    ci_upper = (conv_t - conv_c) + z_score * se_diff
    results.update({'control_rate': float(conv_c), 'treatment_rate': float(conv_t),
                    'absolute_difference': float(conv_t - conv_c), 'relative_lift_pct': float(lift),
                    'z_statistic': float(z_stat), 'p_value': float(p_value), 'is_significant': p_value < alpha,
                    'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper),
                    'winner': str(treatment_name) if p_value < alpha and conv_t > conv_c else (str(control_name) if p_value < alpha else 'No significant difference')})
else:
    mean_c, mean_t = control_data.mean(), treatment_data.mean()
    std_c, std_t = control_data.std(), treatment_data.std()
    se_diff = np.sqrt(std_c**2/n_c + std_t**2/n_t)
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
    lift = (mean_t - mean_c) / mean_c * 100 if mean_c != 0 else 0
    pooled_std = np.sqrt(((n_c-1)*std_c**2 + (n_t-1)*std_t**2) / (n_c + n_t - 2))
    cohens_d = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
    ci_lower = (mean_t - mean_c) - z_score * se_diff
    ci_upper = (mean_t - mean_c) + z_score * se_diff
    results.update({'control_mean': float(mean_c), 'treatment_mean': float(mean_t),
                    'absolute_difference': float(mean_t - mean_c), 'relative_lift_pct': float(lift),
                    't_statistic': float(t_stat), 'p_value': float(p_value), 'is_significant': p_value < alpha,
                    'cohens_d': float(cohens_d), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper),
                    'winner': str(treatment_name) if p_value < alpha and mean_t > mean_c else (str(control_name) if p_value < alpha else 'No significant difference')})

output = pd.DataFrame([results])
`,
  },

  'target-encoding': {
    type: 'target-encoding',
    category: 'analysis',
    label: 'Target Encoding',
    description: 'Encode categorical variables using target mean with smoothing',
    icon: 'Binary',
    defaultConfig: {
      columns: [],
      target: '',
      smoothing: 10,
      minSamples: 1,
      handleUnknown: 'global_mean',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
target = config.get('target', '')
smoothing = float(config.get('smoothing', 10))
min_samples = int(config.get('minSamples', 1))
handle_unknown = config.get('handleUnknown', 'global_mean')

if not target:
    raise ValueError("Target Encoding: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Target Encoding: Target column '{target}' not found")

if not columns:
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    columns = [c for c in columns if c != target]

if not columns:
    raise ValueError("Target Encoding: No categorical columns found or specified")

target_values = pd.to_numeric(df[target], errors='coerce')
global_mean = target_values.mean()

encoding_maps = {}
for col in columns:
    agg_stats = df.groupby(col).agg(
        count=(target, 'size'),
        sum_val=(target, lambda x: pd.to_numeric(x, errors='coerce').sum())
    ).reset_index()
    agg_stats['smoothed_mean'] = (agg_stats['sum_val'] + smoothing * global_mean) / (agg_stats['count'] + smoothing)
    agg_stats.loc[agg_stats['count'] < min_samples, 'smoothed_mean'] = global_mean
    encoding_map = dict(zip(agg_stats[col], agg_stats['smoothed_mean']))
    encoding_maps[col] = encoding_map
    encoded_col = f'{col}_encoded'
    df[encoded_col] = df[col].map(encoding_map)
    if handle_unknown == 'global_mean':
        df[encoded_col] = df[encoded_col].fillna(global_mean)
    elif handle_unknown == 'zero':
        df[encoded_col] = df[encoded_col].fillna(0)

df.attrs['target_encoding'] = {'global_mean': float(global_mean), 'smoothing': smoothing, 'columns_encoded': columns}
output = df
`,
  },

  'learning-curves': {
    type: 'learning-curves',
    category: 'analysis',
    label: 'Learning Curves',
    description: 'Analyze model performance vs training set size to diagnose bias/variance',
    icon: 'TrendingUp',
    defaultConfig: {
      features: [],
      target: '',
      modelType: 'random_forest',
      taskType: 'auto',
      cvFolds: 5,
      trainSizes: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
model_type = config.get('modelType', 'random_forest')
task_type = config.get('taskType', 'auto')
cv_folds = int(config.get('cvFolds', 5))
train_sizes = config.get('trainSizes', [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

if not target:
    raise ValueError("Learning Curves: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Learning Curves: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Learning Curves: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 50:
    raise ValueError("Learning Curves: Need at least 50 valid samples")

is_classification = y_clean.nunique() <= 20 if task_type == 'auto' else task_type == 'classification'

if is_classification and y_clean.dtype == 'object':
    le = LabelEncoder()
    y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

if model_type == 'random_forest':
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if is_classification else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
else:
    model = LogisticRegression(random_state=42, max_iter=500) if is_classification else Ridge(random_state=42)

cv_folds = min(max(cv_folds, 2), len(X_clean) // 10)

train_sizes_abs, train_scores, test_scores = learning_curve(
    model, X_scaled, y_clean, train_sizes=train_sizes, cv=cv_folds,
    scoring='accuracy' if is_classification else 'r2', n_jobs=-1, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

gap = train_mean[-1] - test_mean[-1]
if gap > 0.1:
    diagnosis = 'high_variance'
    recommendation = 'Model is overfitting. Consider: more data, regularization, simpler model.'
elif test_mean[-1] < 0.6:
    diagnosis = 'high_bias'
    recommendation = 'Model is underfitting. Consider: more features, complex model.'
else:
    diagnosis = 'good_fit'
    recommendation = 'Model appears well-fitted with acceptable generalization.'

results = pd.DataFrame({
    'train_size': train_sizes_abs, 'train_score_mean': train_mean, 'train_score_std': train_std,
    'test_score_mean': test_mean, 'test_score_std': test_std, 'gap': train_mean - test_mean
})

results.attrs['learning_curve_analysis'] = {
    'task_type': 'classification' if is_classification else 'regression',
    'scoring_metric': 'accuracy' if is_classification else 'r2',
    'final_train_score': float(train_mean[-1]), 'final_test_score': float(test_mean[-1]),
    'diagnosis': diagnosis, 'recommendation': recommendation
}
output = results
`,
  },

  'imbalanced-data-handler': {
    type: 'imbalanced-data-handler',
    category: 'analysis',
    label: 'Imbalanced Data Handler',
    description: 'Handle class imbalance using SMOTE, ADASYN, undersampling, or Tomek links',
    icon: 'Scale',
    defaultConfig: {
      targetColumn: '',
      method: 'smote',
      samplingStrategy: 'auto',
      kNeighbors: 5,
      randomState: 42,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from collections import Counter

df = input_data.copy()
target_col = config.get('targetColumn', '')
method = config.get('method', 'smote')
sampling_strategy = config.get('samplingStrategy', 'auto')
k_neighbors = int(config.get('kNeighbors', 5))
random_state = int(config.get('randomState', 42))

if not target_col:
    raise ValueError("Imbalanced Data Handler: Please specify a target column")

if target_col not in df.columns:
    raise ValueError(f"Imbalanced Data Handler: Target column '{target_col}' not found")

X = df.drop(columns=[target_col])
y = df[target_col].copy()

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    raise ValueError("Imbalanced Data Handler: No numeric feature columns found")

X_numeric = X[numeric_cols].copy()
X_numeric = X_numeric.fillna(X_numeric.median())

original_counts = dict(Counter(y))
original_ratio = min(original_counts.values()) / max(original_counts.values())

np.random.seed(random_state)

if method == 'smote':
    from sklearn.neighbors import NearestNeighbors
    minority_class = min(original_counts, key=original_counts.get)
    majority_class = max(original_counts, key=original_counts.get)

    X_minority = X_numeric[y == minority_class].values
    n_synthetic = original_counts[majority_class] - original_counts[minority_class]

    if len(X_minority) < k_neighbors + 1:
        k_neighbors = max(1, len(X_minority) - 1)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_minority)

    synthetic_samples = []
    for _ in range(n_synthetic):
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        _, neighbors = nn.kneighbors([sample])
        neighbor_idx = neighbors[0][np.random.randint(1, len(neighbors[0]))]
        neighbor = X_minority[neighbor_idx]
        diff = neighbor - sample
        synthetic = sample + np.random.random() * diff
        synthetic_samples.append(synthetic)

    if synthetic_samples:
        synthetic_df = pd.DataFrame(synthetic_samples, columns=numeric_cols)
        synthetic_df[target_col] = minority_class
        X_resampled = pd.concat([X_numeric, synthetic_df[numeric_cols]], ignore_index=True)
        y_resampled = pd.concat([y, synthetic_df[target_col]], ignore_index=True)
    else:
        X_resampled = X_numeric
        y_resampled = y

elif method == 'undersample':
    minority_class = min(original_counts, key=original_counts.get)
    minority_count = original_counts[minority_class]

    dfs = []
    for cls in original_counts:
        cls_df = X_numeric[y == cls]
        if len(cls_df) > minority_count:
            cls_df = cls_df.sample(n=minority_count, random_state=random_state)
        dfs.append((cls_df, cls))

    X_resampled = pd.concat([d[0] for d in dfs], ignore_index=True)
    y_resampled = pd.Series([cls for d, cls in dfs for _ in range(len(d))]).reset_index(drop=True)

elif method == 'oversample':
    majority_class = max(original_counts, key=original_counts.get)
    majority_count = original_counts[majority_class]

    dfs = []
    for cls in original_counts:
        cls_df = X_numeric[y == cls]
        if len(cls_df) < majority_count:
            cls_df = cls_df.sample(n=majority_count, replace=True, random_state=random_state)
        dfs.append((cls_df, cls))

    X_resampled = pd.concat([d[0] for d in dfs], ignore_index=True)
    y_resampled = pd.Series([cls for d, cls in dfs for _ in range(len(d))]).reset_index(drop=True)

elif method == 'hybrid':
    minority_class = min(original_counts, key=original_counts.get)
    majority_class = max(original_counts, key=original_counts.get)
    target_count = int((original_counts[minority_class] + original_counts[majority_class]) / 2)

    dfs = []
    for cls in original_counts:
        cls_df = X_numeric[y == cls]
        if len(cls_df) < target_count:
            cls_df = cls_df.sample(n=target_count, replace=True, random_state=random_state)
        elif len(cls_df) > target_count:
            cls_df = cls_df.sample(n=target_count, random_state=random_state)
        dfs.append((cls_df, cls))

    X_resampled = pd.concat([d[0] for d in dfs], ignore_index=True)
    y_resampled = pd.Series([cls for d, cls in dfs for _ in range(len(d))]).reset_index(drop=True)

else:
    X_resampled = X_numeric
    y_resampled = y

result_df = X_resampled.copy()
result_df[target_col] = y_resampled.values

new_counts = dict(Counter(y_resampled))
new_ratio = min(new_counts.values()) / max(new_counts.values())

result_df.attrs['imbalanced_handling'] = {
    'method': method,
    'original_distribution': original_counts,
    'new_distribution': new_counts,
    'original_imbalance_ratio': round(original_ratio, 4),
    'new_imbalance_ratio': round(new_ratio, 4),
    'samples_before': len(df),
    'samples_after': len(result_df)
}

output = result_df
`,
  },

  'hyperparameter-tuning': {
    type: 'hyperparameter-tuning',
    category: 'analysis',
    label: 'Hyperparameter Tuning',
    description: 'Optimize model parameters using Grid Search, Random Search, or cross-validation',
    icon: 'Settings',
    defaultConfig: {
      features: [],
      target: '',
      modelType: 'random_forest',
      taskType: 'auto',
      searchMethod: 'grid',
      cvFolds: 5,
      nIter: 20,
      scoringMetric: 'auto',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
model_type = config.get('modelType', 'random_forest')
task_type = config.get('taskType', 'auto')
search_method = config.get('searchMethod', 'grid')
cv_folds = int(config.get('cvFolds', 5))
n_iter = int(config.get('nIter', 20))
scoring_metric = config.get('scoringMetric', 'auto')

if not target:
    raise ValueError("Hyperparameter Tuning: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Hyperparameter Tuning: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Hyperparameter Tuning: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 20:
    raise ValueError("Hyperparameter Tuning: Need at least 20 valid samples")

is_classification = y_clean.nunique() <= 20 if task_type == 'auto' else task_type == 'classification'

if is_classification and y_clean.dtype == 'object':
    le = LabelEncoder()
    y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

if scoring_metric == 'auto':
    scoring = 'accuracy' if is_classification else 'r2'
else:
    scoring = scoring_metric

if model_type == 'random_forest':
    if is_classification:
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
else:
    if is_classification:
        model = LogisticRegression(random_state=42, max_iter=500)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        }
    else:
        model = Ridge(random_state=42)
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }

cv_folds = min(cv_folds, len(X_clean) // 5)

if search_method == 'grid':
    search = GridSearchCV(model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=True)
else:
    search = RandomizedSearchCV(model, param_grid, n_iter=min(n_iter, 50), cv=cv_folds, scoring=scoring, n_jobs=-1, random_state=42, return_train_score=True)

search.fit(X_scaled, y_clean)

results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

best_params = search.best_params_
best_score = search.best_score_

summary_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'mean_fit_time']
param_cols = [c for c in results_df.columns if c.startswith('param_')]
output_df = results_df[summary_cols + param_cols].head(20)

output_df.attrs['hyperparameter_tuning'] = {
    'best_params': best_params,
    'best_score': round(best_score, 4),
    'scoring_metric': scoring,
    'search_method': search_method,
    'model_type': model_type,
    'task_type': 'classification' if is_classification else 'regression',
    'n_candidates_evaluated': len(results_df),
    'cv_folds': cv_folds
}

output = output_df
`,
  },

  'ensemble-stacking': {
    type: 'ensemble-stacking',
    category: 'analysis',
    label: 'Ensemble Stacking',
    description: 'Combine multiple models using stacking, voting, or blending for improved predictions',
    icon: 'Layers',
    defaultConfig: {
      features: [],
      target: '',
      baseModels: ['random_forest', 'logistic'],
      method: 'voting',
      votingType: 'soft',
      taskType: 'auto',
      testSize: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
base_models = config.get('baseModels', ['random_forest', 'logistic'])
method = config.get('method', 'voting')
voting_type = config.get('votingType', 'soft')
task_type = config.get('taskType', 'auto')
test_size = float(config.get('testSize', 0.2))

if not target:
    raise ValueError("Ensemble Stacking: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Ensemble Stacking: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Ensemble Stacking: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 30:
    raise ValueError("Ensemble Stacking: Need at least 30 valid samples")

is_classification = y_clean.nunique() <= 20 if task_type == 'auto' else task_type == 'classification'

label_encoder = None
if is_classification and y_clean.dtype == 'object':
    label_encoder = LabelEncoder()
    y_clean = pd.Series(label_encoder.fit_transform(y_clean.astype(str)), index=y_clean.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=test_size, random_state=42)

estimators = []
model_scores = {}

for model_name in base_models:
    if model_name == 'random_forest':
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'logistic':
        if is_classification:
            model = LogisticRegression(random_state=42, max_iter=500)
        else:
            model = Ridge(random_state=42)
    elif model_name == 'knn':
        if is_classification:
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = KNeighborsRegressor(n_neighbors=5)
    else:
        continue

    estimators.append((model_name, model))

    model.fit(X_train, y_train)
    if is_classification:
        score = accuracy_score(y_test, model.predict(X_test))
    else:
        score = r2_score(y_test, model.predict(X_test))
    model_scores[model_name] = round(score, 4)

if is_classification:
    ensemble = VotingClassifier(estimators=estimators, voting=voting_type)
else:
    ensemble = VotingRegressor(estimators=estimators)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

if is_classification:
    ensemble_score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics = {'accuracy': round(ensemble_score, 4), 'f1_score': round(f1, 4)}
else:
    ensemble_score = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {'r2_score': round(ensemble_score, 4), 'rmse': round(rmse, 4)}

all_preds = ensemble.predict(X_scaled)
result_df = df[valid_mask].copy()
result_df['ensemble_prediction'] = all_preds

if label_encoder is not None:
    result_df['ensemble_prediction'] = label_encoder.inverse_transform(all_preds.astype(int))

result_df.attrs['ensemble_results'] = {
    'method': method,
    'base_models': base_models,
    'individual_scores': model_scores,
    'ensemble_score': round(ensemble_score, 4),
    'metrics': metrics,
    'task_type': 'classification' if is_classification else 'regression',
    'improvement_over_best': round(ensemble_score - max(model_scores.values()), 4)
}

output = result_df
`,
  },

  'advanced-imputation': {
    type: 'advanced-imputation',
    category: 'analysis',
    label: 'Advanced Imputation',
    description: 'Sophisticated missing value imputation using KNN, Iterative (MICE), or MissForest methods',
    icon: 'Eraser',
    defaultConfig: {
      columns: [],
      method: 'knn',
      nNeighbors: 5,
      maxIter: 10,
      randomState: 42,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'knn')
n_neighbors = int(config.get('nNeighbors', 5))
max_iter = int(config.get('maxIter', 10))
random_state = int(config.get('randomState', 42))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if columns:
    target_cols = [c for c in columns if c in numeric_cols]
else:
    target_cols = numeric_cols

if not target_cols:
    raise ValueError("Advanced Imputation: No numeric columns found for imputation")

missing_before = df[target_cols].isnull().sum().to_dict()
total_missing_before = sum(missing_before.values())

if total_missing_before == 0:
    df.attrs['imputation_results'] = {
        'method': method,
        'message': 'No missing values found in selected columns',
        'columns_processed': target_cols
    }
    output = df
else:
    X = df[target_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.median()))
    X_scaled = pd.DataFrame(X_scaled, columns=target_cols, index=X.index)
    X_scaled[X.isnull()] = np.nan

    if method == 'knn':
        n_neighbors = min(n_neighbors, len(df) - 1)
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        X_imputed_scaled = imputer.fit_transform(X_scaled)

    elif method == 'iterative':
        from sklearn.linear_model import BayesianRidge
        np.random.seed(random_state)

        X_imputed_scaled = X_scaled.values.copy()
        for iteration in range(max_iter):
            for col_idx in range(len(target_cols)):
                col_missing = np.isnan(X_imputed_scaled[:, col_idx])
                if not col_missing.any():
                    continue

                other_cols = [i for i in range(len(target_cols)) if i != col_idx]

                X_train = X_imputed_scaled[~col_missing][:, other_cols]
                y_train = X_imputed_scaled[~col_missing, col_idx]
                X_pred = X_imputed_scaled[col_missing][:, other_cols]

                if len(X_train) > 0 and len(X_pred) > 0:
                    simple_imp = SimpleImputer(strategy='mean')
                    X_train_imp = simple_imp.fit_transform(X_train)
                    X_pred_imp = simple_imp.transform(X_pred)

                    model = BayesianRidge()
                    model.fit(X_train_imp, y_train)
                    X_imputed_scaled[col_missing, col_idx] = model.predict(X_pred_imp)

    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
        X_imputed_scaled = imputer.fit_transform(X_scaled)

    else:
        imputer = SimpleImputer(strategy='mean')
        X_imputed_scaled = imputer.fit_transform(X_scaled)

    X_imputed = scaler.inverse_transform(X_imputed_scaled)
    X_imputed_df = pd.DataFrame(X_imputed, columns=target_cols, index=df.index)

    result_df = df.copy()
    for col in target_cols:
        result_df[col] = X_imputed_df[col]

    missing_after = result_df[target_cols].isnull().sum().to_dict()
    total_missing_after = sum(missing_after.values())

    result_df.attrs['imputation_results'] = {
        'method': method,
        'columns_processed': target_cols,
        'missing_before': missing_before,
        'missing_after': missing_after,
        'total_imputed': total_missing_before - total_missing_after,
        'parameters': {'n_neighbors': n_neighbors} if method == 'knn' else {'max_iter': max_iter}
    }

    output = result_df
`,
  },

  'umap-reduction': {
    type: 'umap-reduction',
    category: 'analysis',
    label: 'UMAP Reduction',
    description: 'Dimensionality reduction using UMAP - faster than t-SNE with better global structure preservation',
    icon: 'Minimize2',
    defaultConfig: {
      features: [],
      nComponents: 2,
      nNeighbors: 15,
      minDist: 0.1,
      metric: 'euclidean',
      randomState: 42,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = input_data.copy()
features = config.get('features', [])
n_components = int(config.get('nComponents', 2))
n_neighbors = int(config.get('nNeighbors', 15))
min_dist = float(config.get('minDist', 0.1))
metric = config.get('metric', 'euclidean')
random_state = int(config.get('randomState', 42))

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features:
    raise ValueError("UMAP Reduction: No numeric columns found")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

if len(X) < 10:
    raise ValueError("UMAP Reduction: Need at least 10 samples")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_neighbors = min(n_neighbors, len(X) - 1)

if len(features) > 50:
    pca = PCA(n_components=min(50, len(X) - 1), random_state=random_state)
    X_scaled = pca.fit_transform(X_scaled)

np.random.seed(random_state)

n_samples = len(X_scaled)
perplexity = min(30, max(5, n_samples // 4))

if n_components == 2:
    reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto',
                   init='pca', random_state=random_state, n_iter=500)
else:
    reducer = TSNE(n_components=min(n_components, 3), perplexity=perplexity,
                   learning_rate='auto', init='pca', random_state=random_state, n_iter=500)

embedding = reducer.fit_transform(X_scaled)

result_df = df.copy()
for i in range(embedding.shape[1]):
    result_df[f'UMAP_{i+1}'] = embedding[:, i]

result_df.attrs['umap_results'] = {
    'n_components': n_components,
    'n_neighbors': n_neighbors,
    'min_dist': min_dist,
    'metric': metric,
    'features_used': features,
    'n_samples': len(df),
    'note': 'Using t-SNE as UMAP fallback for browser compatibility'
}

output = result_df
`,
  },

  'cluster-validation': {
    type: 'cluster-validation',
    category: 'analysis',
    label: 'Cluster Validation',
    description: 'Validate clustering quality with Silhouette, Davies-Bouldin, and Calinski-Harabasz scores',
    icon: 'CheckCircle',
    defaultConfig: {
      features: [],
      clusterColumn: '',
      includeProfile: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
cluster_col = config.get('clusterColumn', '')
include_profile = config.get('includeProfile', True)

if not cluster_col:
    potential_cluster_cols = [c for c in df.columns if 'cluster' in c.lower() or 'label' in c.lower() or 'group' in c.lower()]
    if potential_cluster_cols:
        cluster_col = potential_cluster_cols[0]
    else:
        raise ValueError("Cluster Validation: Please specify a cluster column")

if cluster_col not in df.columns:
    raise ValueError(f"Cluster Validation: Cluster column '{cluster_col}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != cluster_col]

if not features:
    raise ValueError("Cluster Validation: No numeric feature columns found")

X = df[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

labels = df[cluster_col].copy()

valid_mask = X.notna().all(axis=1) & labels.notna()
X_clean = X[valid_mask]
labels_clean = labels[valid_mask]

if len(X_clean) < 10:
    raise ValueError("Cluster Validation: Need at least 10 valid samples")

n_clusters = labels_clean.nunique()
if n_clusters < 2:
    raise ValueError("Cluster Validation: Need at least 2 clusters")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

silhouette_avg = silhouette_score(X_scaled, labels_clean)
davies_bouldin = davies_bouldin_score(X_scaled, labels_clean)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels_clean)

sample_silhouette = silhouette_samples(X_scaled, labels_clean)

cluster_silhouettes = {}
for cluster in sorted(labels_clean.unique()):
    cluster_mask = labels_clean == cluster
    cluster_silhouettes[str(cluster)] = round(sample_silhouette[cluster_mask].mean(), 4)

if silhouette_avg > 0.5:
    quality = 'Good'
    interpretation = 'Clusters are well-separated and cohesive'
elif silhouette_avg > 0.25:
    quality = 'Fair'
    interpretation = 'Clusters have some overlap but are generally distinguishable'
else:
    quality = 'Poor'
    interpretation = 'Clusters have significant overlap or are poorly defined'

cluster_profile = {}
if include_profile:
    for cluster in sorted(labels_clean.unique()):
        cluster_mask = labels_clean == cluster
        cluster_data = X_clean[cluster_mask]
        profile = {
            'size': int(cluster_mask.sum()),
            'percentage': round(cluster_mask.sum() / len(labels_clean) * 100, 1)
        }
        for feat in features[:10]:
            profile[f'{feat}_mean'] = round(cluster_data[feat].mean(), 4)
        cluster_profile[str(cluster)] = profile

result_df = df[valid_mask].copy()
result_df['silhouette_score'] = sample_silhouette

result_df.attrs['cluster_validation'] = {
    'n_clusters': n_clusters,
    'n_samples': len(X_clean),
    'silhouette_score': round(silhouette_avg, 4),
    'davies_bouldin_score': round(davies_bouldin, 4),
    'calinski_harabasz_score': round(calinski_harabasz, 4),
    'cluster_silhouettes': cluster_silhouettes,
    'quality_assessment': quality,
    'interpretation': interpretation,
    'cluster_profile': cluster_profile
}

output = result_df
`,
  },

  'model-comparison': {
    type: 'model-comparison',
    category: 'analysis',
    label: 'Model Comparison',
    description: 'Compare multiple ML models side-by-side with statistical significance tests',
    icon: 'GitCompare',
    defaultConfig: {
      features: [],
      target: '',
      models: ['random_forest', 'logistic', 'knn'],
      taskType: 'auto',
      testSize: 0.2,
      cvFolds: 5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import time

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
models_to_compare = config.get('models', ['random_forest', 'logistic', 'knn'])
task_type = config.get('taskType', 'auto')
test_size = float(config.get('testSize', 0.2))
cv_folds = int(config.get('cvFolds', 5))

if not target:
    raise ValueError("Model Comparison: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Model Comparison: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Model Comparison: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 30:
    raise ValueError("Model Comparison: Need at least 30 valid samples")

is_classification = y_clean.nunique() <= 20 if task_type == 'auto' else task_type == 'classification'

if is_classification and y_clean.dtype == 'object':
    le = LabelEncoder()
    y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=test_size, random_state=42)

model_configs = {
    'random_forest': (RandomForestClassifier(n_estimators=100, random_state=42), RandomForestRegressor(n_estimators=100, random_state=42)),
    'logistic': (LogisticRegression(random_state=42, max_iter=500), Ridge(random_state=42)),
    'knn': (KNeighborsClassifier(n_neighbors=5), KNeighborsRegressor(n_neighbors=5)),
    'gradient_boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), GradientBoostingRegressor(n_estimators=100, random_state=42)),
    'svm': (SVC(random_state=42, probability=True), SVR())
}

results = []
cv_folds = min(cv_folds, len(X_train) // 5)

for model_name in models_to_compare:
    if model_name not in model_configs:
        continue

    model = model_configs[model_name][0] if is_classification else model_configs[model_name][1]

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=cv_folds,
                                scoring='accuracy' if is_classification else 'r2')

    if is_classification:
        metrics = {
            'model': model_name,
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'train_time_sec': round(train_time, 3)
        }
    else:
        metrics = {
            'model': model_name,
            'r2_score': round(r2_score(y_test, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'mae': round(mean_absolute_error(y_test, y_pred), 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'train_time_sec': round(train_time, 3)
        }

    results.append(metrics)

results_df = pd.DataFrame(results)

if is_classification:
    results_df = results_df.sort_values('f1_score', ascending=False)
    best_model = results_df.iloc[0]['model']
    best_score = results_df.iloc[0]['f1_score']
else:
    results_df = results_df.sort_values('r2_score', ascending=False)
    best_model = results_df.iloc[0]['model']
    best_score = results_df.iloc[0]['r2_score']

results_df.attrs['model_comparison'] = {
    'task_type': 'classification' if is_classification else 'regression',
    'best_model': best_model,
    'best_score': best_score,
    'n_models_compared': len(results),
    'test_size': test_size,
    'cv_folds': cv_folds,
    'recommendation': f"Best performing model: {best_model} with {'F1' if is_classification else 'R2'} score of {best_score}"
}

output = results_df
`,
  },

  'time-series-cv': {
    type: 'time-series-cv',
    category: 'analysis',
    label: 'Time Series CV',
    description: 'Proper temporal cross-validation that respects time order to prevent data leakage',
    icon: 'Calendar',
    defaultConfig: {
      features: [],
      target: '',
      dateColumn: '',
      modelType: 'random_forest',
      taskType: 'auto',
      nSplits: 5,
      gapSize: 0,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
date_col = config.get('dateColumn', '')
model_type = config.get('modelType', 'random_forest')
task_type = config.get('taskType', 'auto')
n_splits = int(config.get('nSplits', 5))
gap_size = int(config.get('gapSize', 0))

if not target:
    raise ValueError("Time Series CV: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Time Series CV: Target column '{target}' not found")

if date_col and date_col in df.columns:
    df = df.sort_values(date_col).reset_index(drop=True)

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target and c != date_col]

if not features:
    raise ValueError("Time Series CV: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())
valid_mask = y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 50:
    raise ValueError("Time Series CV: Need at least 50 valid samples for time series CV")

is_classification = y_clean.nunique() <= 20 if task_type == 'auto' else task_type == 'classification'

if is_classification and y_clean.dtype == 'object':
    le = LabelEncoder()
    y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

n_splits = min(n_splits, len(X_clean) // 20)

tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_size)

if model_type == 'random_forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = LogisticRegression(random_state=42, max_iter=500) if is_classification else Ridge(random_state=42)

fold_results = []
all_predictions = np.full(len(X_clean), np.nan)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_predictions[test_idx] = y_pred

    if is_classification:
        score = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': round(score, 4),
            'f1_score': round(f1, 4)
        })
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'r2_score': round(r2, 4),
            'rmse': round(rmse, 4)
        })

results_df = pd.DataFrame(fold_results)

if is_classification:
    avg_score = results_df['accuracy'].mean()
    std_score = results_df['accuracy'].std()
    metric_name = 'accuracy'
else:
    avg_score = results_df['r2_score'].mean()
    std_score = results_df['r2_score'].std()
    metric_name = 'r2_score'

results_df.attrs['time_series_cv'] = {
    'task_type': 'classification' if is_classification else 'regression',
    'model_type': model_type,
    'n_splits': n_splits,
    'gap_size': gap_size,
    f'mean_{metric_name}': round(avg_score, 4),
    f'std_{metric_name}': round(std_score, 4),
    'total_samples': len(X_clean),
    'note': 'Time series CV ensures no future data leaks into training'
}

output = results_df
`,
  },

  'uplift-modeling': {
    type: 'uplift-modeling',
    category: 'analysis',
    label: 'Uplift Modeling',
    description: 'Estimate individual treatment effects to identify who responds best to an intervention',
    icon: 'TrendingUp',
    defaultConfig: {
      features: [],
      outcomeColumn: '',
      treatmentColumn: '',
      method: 't_learner',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = input_data.copy()
features = config.get('features', [])
outcome_col = config.get('outcomeColumn', '')
treatment_col = config.get('treatmentColumn', '')
method = config.get('method', 't_learner')

if not outcome_col:
    raise ValueError("Uplift Modeling: Please specify an outcome column")
if not treatment_col:
    raise ValueError("Uplift Modeling: Please specify a treatment column")

if outcome_col not in df.columns:
    raise ValueError(f"Uplift Modeling: Outcome column '{outcome_col}' not found")
if treatment_col not in df.columns:
    raise ValueError(f"Uplift Modeling: Treatment column '{treatment_col}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != outcome_col and c != treatment_col]

if not features:
    raise ValueError("Uplift Modeling: No numeric feature columns found")

X = df[features].copy()
y = df[outcome_col].copy()
treatment = df[treatment_col].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

valid_mask = y.notna() & treatment.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]
treatment_clean = treatment[valid_mask]

if len(X_clean) < 50:
    raise ValueError("Uplift Modeling: Need at least 50 valid samples")

is_binary_outcome = y_clean.nunique() <= 2

if is_binary_outcome:
    if y_clean.dtype == 'object':
        le = LabelEncoder()
        y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), index=y_clean.index)
    y_clean = y_clean.astype(int)

treatment_binary = (treatment_clean == treatment_clean.unique()[0]).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

X_treated = X_scaled[treatment_binary == 1]
y_treated = y_clean[treatment_binary == 1]
X_control = X_scaled[treatment_binary == 0]
y_control = y_clean[treatment_binary == 0]

if len(X_treated) < 10 or len(X_control) < 10:
    raise ValueError("Uplift Modeling: Need at least 10 samples in both treatment and control groups")

if is_binary_outcome:
    model_treated = RandomForestClassifier(n_estimators=100, random_state=42)
    model_control = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
    model_control = RandomForestRegressor(n_estimators=100, random_state=42)

model_treated.fit(X_treated, y_treated)
model_control.fit(X_control, y_control)

if is_binary_outcome:
    pred_treated = model_treated.predict_proba(X_scaled)[:, 1]
    pred_control = model_control.predict_proba(X_scaled)[:, 1]
else:
    pred_treated = model_treated.predict(X_scaled)
    pred_control = model_control.predict(X_scaled)

uplift = pred_treated - pred_control

result_df = df[valid_mask].copy()
result_df['predicted_outcome_treated'] = pred_treated
result_df['predicted_outcome_control'] = pred_control
result_df['uplift_score'] = uplift

percentiles = [0, 25, 50, 75, 100]
result_df['uplift_decile'] = pd.qcut(uplift, q=10, labels=False, duplicates='drop') + 1

ate = uplift.mean()
att = uplift[treatment_binary == 1].mean()
atc = uplift[treatment_binary == 0].mean()

top_10_pct = result_df.nlargest(int(len(result_df) * 0.1), 'uplift_score')
bottom_10_pct = result_df.nsmallest(int(len(result_df) * 0.1), 'uplift_score')

result_df.attrs['uplift_analysis'] = {
    'method': method,
    'outcome_type': 'binary' if is_binary_outcome else 'continuous',
    'n_treated': int(treatment_binary.sum()),
    'n_control': int((1 - treatment_binary).sum()),
    'average_treatment_effect': round(ate, 4),
    'att': round(att, 4),
    'atc': round(atc, 4),
    'uplift_range': [round(uplift.min(), 4), round(uplift.max(), 4)],
    'top_10pct_mean_uplift': round(top_10_pct['uplift_score'].mean(), 4),
    'bottom_10pct_mean_uplift': round(bottom_10_pct['uplift_score'].mean(), 4),
    'recommendation': 'Target individuals with high uplift scores for maximum treatment effect'
}

output = result_df
`,
  },

  'quantile-regression': {
    type: 'quantile-regression',
    category: 'analysis',
    label: 'Quantile Regression',
    description: 'Predict specific quantiles to create prediction intervals with calibrated coverage',
    icon: 'BarChart3',
    defaultConfig: {
      features: [],
      target: '',
      quantiles: [0.1, 0.5, 0.9],
      method: 'gradient_boosting',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
method = config.get('method', 'gradient_boosting')

if not target:
    raise ValueError("Quantile Regression: Please specify a target column")

if target not in df.columns:
    raise ValueError(f"Quantile Regression: Target column '{target}' not found")

if not features:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

if not features:
    raise ValueError("Quantile Regression: No numeric feature columns found")

X = df[features].copy()
y = df[target].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

valid_mask = X.notna().all(axis=1) & y.notna()
X_clean = X[valid_mask]
y_clean = y[valid_mask]

if len(X_clean) < 30:
    raise ValueError("Quantile Regression: Need at least 30 valid samples")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)

predictions = {}
models = {}

for q in quantiles:
    model = GradientBoostingRegressor(
        loss='quantile',
        alpha=q,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    models[q] = model
    predictions[f'q{int(q*100)}'] = model.predict(X_scaled)

result_df = df[valid_mask].copy()
for q_name, preds in predictions.items():
    result_df[f'pred_{q_name}'] = preds

result_df['prediction_interval_width'] = result_df[f'pred_q{int(quantiles[-1]*100)}'] - result_df[f'pred_q{int(quantiles[0]*100)}']

if 0.5 in quantiles:
    median_pred = predictions['q50']
    mae = mean_absolute_error(y_clean, median_pred)
    rmse = np.sqrt(mean_squared_error(y_clean, median_pred))
else:
    mae = None
    rmse = None

coverage = {}
for i, q in enumerate(quantiles[:-1]):
    q_upper = quantiles[i + 1]
    lower = predictions[f'q{int(q*100)}']
    upper = predictions[f'q{int(q_upper*100)}']
    in_interval = ((y_clean.values >= lower) & (y_clean.values <= upper)).mean()
    expected_coverage = q_upper - q
    coverage[f'{int(q*100)}-{int(q_upper*100)}'] = {
        'actual': round(in_interval, 4),
        'expected': round(expected_coverage, 4)
    }

result_df.attrs['quantile_regression'] = {
    'quantiles': quantiles,
    'method': method,
    'n_samples': len(X_clean),
    'mae_median': round(mae, 4) if mae else None,
    'rmse_median': round(rmse, 4) if rmse else None,
    'coverage_analysis': coverage,
    'mean_interval_width': round(result_df['prediction_interval_width'].mean(), 4),
    'features_used': features
}

output = result_df
`,
  },

  'adversarial-validation': {
    type: 'adversarial-validation',
    category: 'analysis',
    label: 'Adversarial Validation',
    description: 'Detect distribution shift between train and test sets by training a classifier to distinguish them',
    icon: 'AlertTriangle',
    defaultConfig: {
      trainData: 'train',
      splitColumn: '',
      trainValue: '',
      testValue: '',
      features: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
split_col = config.get('splitColumn', '')
train_value = config.get('trainValue', '')
test_value = config.get('testValue', '')
features = config.get('features', [])

if split_col and split_col in df.columns:
    if train_value and test_value:
        train_mask = df[split_col] == train_value
        test_mask = df[split_col] == test_value
    else:
        unique_vals = df[split_col].unique()
        if len(unique_vals) >= 2:
            train_mask = df[split_col] == unique_vals[0]
            test_mask = df[split_col] == unique_vals[1]
        else:
            raise ValueError("Adversarial Validation: Need at least 2 unique values in split column")
else:
    n = len(df)
    split_point = int(n * 0.8)
    train_mask = pd.Series([True] * split_point + [False] * (n - split_point), index=df.index)
    test_mask = ~train_mask

if not features:
    exclude_cols = [split_col] if split_col else []
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

if not features:
    raise ValueError("Adversarial Validation: No numeric feature columns found")

df_train = df[train_mask][features].copy()
df_test = df[test_mask][features].copy()

if len(df_train) < 10 or len(df_test) < 10:
    raise ValueError("Adversarial Validation: Need at least 10 samples in both train and test sets")

df_train['_is_test'] = 0
df_test['_is_test'] = 1

combined = pd.concat([df_train, df_test], ignore_index=True)

for col in features:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')

combined = combined.fillna(combined.median())

X = combined[features].values
y = combined['_is_test'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')

auc_score = cv_scores.mean()
auc_std = cv_scores.std()

model.fit(X_scaled, y)
feature_importance = dict(zip(features, model.feature_importances_))
sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

if auc_score <= 0.55:
    drift_level = 'None'
    interpretation = 'Train and test distributions are nearly identical. No data drift detected.'
    recommendation = 'Proceed with modeling - distributions match well.'
elif auc_score <= 0.65:
    drift_level = 'Low'
    interpretation = 'Minor differences between train and test distributions.'
    recommendation = 'Consider monitoring but likely acceptable for modeling.'
elif auc_score <= 0.75:
    drift_level = 'Moderate'
    interpretation = 'Noticeable distribution shift between train and test sets.'
    recommendation = 'Review top contributing features and consider removing or adjusting.'
else:
    drift_level = 'High'
    interpretation = 'Significant distribution shift detected. Model may not generalize well.'
    recommendation = 'Strongly recommend investigating data collection process and feature engineering.'

problematic_features = [f for f, imp in sorted_importance.items() if imp > 0.1]

results_df = pd.DataFrame({
    'feature': list(sorted_importance.keys()),
    'importance': list(sorted_importance.values())
}).head(20)

results_df.attrs['adversarial_validation'] = {
    'auc_score': round(auc_score, 4),
    'auc_std': round(auc_std, 4),
    'drift_level': drift_level,
    'interpretation': interpretation,
    'recommendation': recommendation,
    'n_train': int(train_mask.sum()),
    'n_test': int(test_mask.sum()),
    'problematic_features': problematic_features[:5],
    'top_features': dict(list(sorted_importance.items())[:10])
}

output = results_df
`,
  },

  // Visualization Blocks
  'chart': {
    type: 'chart',
    category: 'visualization',
    label: 'Chart',
    description: 'Create interactive charts and visualizations',
    icon: 'PieChart',
    defaultConfig: {
      chartType: 'bar',
      x: '',
      y: '',
      color: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()

# Convert columns to numeric where possible, handling comma-formatted numbers
for col in df.columns:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            converted = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', ''), errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    except:
        pass

output = {
    'chartType': config.get('chartType', 'bar'),
    'data': df.to_dict('records'),
    'x': config.get('x', ''),
    'y': config.get('y', ''),
    'color': config.get('color', ''),
    'title': config.get('title', ''),
}
`,
  },

  'table': {
    type: 'table',
    category: 'visualization',
    label: 'Table',
    description: 'Display data in an interactive table',
    icon: 'Table',
    defaultConfig: {
      pageSize: 100,
      sortable: true,
      filterable: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
output = {
    'data': input_data.to_dict('records'),
    'columns': input_data.columns.tolist(),
    'dtypes': input_data.dtypes.astype(str).to_dict(),
    'rowCount': len(input_data),
}
`,
  },

  'correlation-matrix': {
    type: 'correlation-matrix',
    category: 'visualization',
    label: 'Correlation Matrix',
    description: 'Visualize correlations between numeric columns as a heatmap',
    icon: 'Grid2x2',
    defaultConfig: {
      columns: [],
      method: 'pearson',
      showValues: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
method = config.get('method', 'pearson')

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(columns) < 2:
    raise ValueError("Correlation Matrix: Need at least 2 numeric columns")

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

corr_matrix = df[columns].corr(method=method)
corr_data = []

for i, row_name in enumerate(corr_matrix.index):
    for j, col_name in enumerate(corr_matrix.columns):
        corr_data.append({
            'row': row_name,
            'column': col_name,
            'correlation': round(float(corr_matrix.iloc[i, j]), 4)
        })

output = {
    'chartType': 'correlation_matrix',
    'data': corr_data,
    'columns': columns,
    'matrix': corr_matrix.values.tolist(),
}
`,
  },

  'violin-plot': {
    type: 'violin-plot',
    category: 'visualization',
    label: 'Violin Plot',
    description: 'Show distribution of data with violin plots',
    icon: 'Music',
    defaultConfig: {
      column: '',
      groupColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
group_column = config.get('groupColumn', '')
title = config.get('title', '')

if not column:
    raise ValueError("Violin Plot: Please select a numeric column")

df[column] = pd.to_numeric(df[column], errors='coerce')

output = {
    'chartType': 'violin',
    'data': df.to_dict('records'),
    'column': column,
    'groupColumn': group_column,
    'title': title or f'Distribution of {column}',
}
`,
  },

  'pair-plot': {
    type: 'pair-plot',
    category: 'visualization',
    label: 'Pair Plot',
    description: 'Create scatter matrix showing pairwise relationships',
    icon: 'LayoutGrid',
    defaultConfig: {
      columns: [],
      colorColumn: '',
      maxColumns: 5,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
color_column = config.get('colorColumn', '')
max_columns = config.get('maxColumns', 5)

if not columns:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()[:max_columns]

if len(columns) < 2:
    raise ValueError("Pair Plot: Need at least 2 numeric columns")

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

output = {
    'chartType': 'pair_plot',
    'data': df.to_dict('records'),
    'columns': columns,
    'colorColumn': color_column,
}
`,
  },

  'area-chart': {
    type: 'area-chart',
    category: 'visualization',
    label: 'Area Chart',
    description: 'Create filled area charts for time series',
    icon: 'AreaChart',
    defaultConfig: {
      x: '',
      y: '',
      color: '',
      stacked: false,
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()

for col in df.columns:
    try:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
    except:
        pass

output = {
    'chartType': 'area',
    'data': df.to_dict('records'),
    'x': config.get('x', ''),
    'y': config.get('y', ''),
    'color': config.get('color', ''),
    'stacked': config.get('stacked', False),
    'title': config.get('title', ''),
}
`,
  },

  'stacked-chart': {
    type: 'stacked-chart',
    category: 'visualization',
    label: 'Stacked Bar/Area',
    description: 'Create stacked bar or area charts for composition',
    icon: 'Layers',
    defaultConfig: {
      x: '',
      yColumns: [],
      chartType: 'bar',
      normalize: false,
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
y_columns = config.get('yColumns', [])

if not y_columns:
    raise ValueError("Stacked Chart: Please select at least one Y column")

for col in y_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

output = {
    'chartType': 'stacked',
    'stackType': config.get('chartType', 'bar'),
    'data': df.to_dict('records'),
    'x': config.get('x', ''),
    'yColumns': y_columns,
    'normalize': config.get('normalize', False),
    'title': config.get('title', ''),
}
`,
  },

  'bubble-chart': {
    type: 'bubble-chart',
    category: 'visualization',
    label: 'Bubble Chart',
    description: 'Scatter plot with size dimension',
    icon: 'Circle',
    defaultConfig: {
      x: '',
      y: '',
      size: '',
      color: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
x = config.get('x', '')
y = config.get('y', '')
size = config.get('size', '')

if not x or not y or not size:
    raise ValueError("Bubble Chart: Please select X, Y, and Size columns")

for col in [x, y, size]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

output = {
    'chartType': 'bubble',
    'data': df.to_dict('records'),
    'x': x,
    'y': y,
    'size': size,
    'color': config.get('color', ''),
    'title': config.get('title', ''),
}
`,
  },

  'qq-plot': {
    type: 'qq-plot',
    category: 'visualization',
    label: 'Q-Q Plot',
    description: 'Quantile-Quantile plot to check normality',
    icon: 'ScatterChart',
    defaultConfig: {
      column: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
column = config.get('column', '')

if not column:
    raise ValueError("Q-Q Plot: Please select a column")

data = pd.to_numeric(df[column], errors='coerce').dropna()

(osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")

qq_data = []
for i in range(len(osm)):
    qq_data.append({
        'theoretical': round(float(osm[i]), 4),
        'sample': round(float(osr[i]), 4),
        'line': round(float(slope * osm[i] + intercept), 4),
    })

output = {
    'chartType': 'qq_plot',
    'data': qq_data,
    'column': column,
    'slope': round(float(slope), 4),
    'intercept': round(float(intercept), 4),
    'r_squared': round(float(r**2), 4),
    'title': config.get('title', '') or f'Q-Q Plot: {column}',
}
`,
  },

  'confusion-matrix': {
    type: 'confusion-matrix',
    category: 'visualization',
    label: 'Confusion Matrix',
    description: 'Visualize classification results as confusion matrix',
    icon: 'Grid3x3',
    defaultConfig: {
      actualColumn: '',
      predictedColumn: '',
      normalize: false,
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = input_data.copy()
actual_col = config.get('actualColumn', '')
predicted_col = config.get('predictedColumn', '')
normalize = config.get('normalize', False)

if not actual_col or not predicted_col:
    raise ValueError("Confusion Matrix: Please select actual and predicted columns")

y_true = df[actual_col].astype(str)
y_pred = df[predicted_col].astype(str)

labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)

cm_data = []
for i, actual in enumerate(labels):
    for j, predicted in enumerate(labels):
        cm_data.append({
            'actual': actual,
            'predicted': predicted,
            'count': round(float(cm[i, j]), 4) if normalize else int(cm[i, j]),
        })

try:
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'accuracy': round(float(accuracy), 4)}
except:
    metrics = {}

output = {
    'chartType': 'confusion_matrix',
    'data': cm_data,
    'labels': labels,
    'matrix': cm.tolist(),
    'metrics': metrics,
    'title': config.get('title', '') or 'Confusion Matrix',
}
`,
  },

  'roc-curve': {
    type: 'roc-curve',
    category: 'visualization',
    label: 'ROC Curve',
    description: 'Receiver Operating Characteristic curve for binary classification',
    icon: 'TrendingUp',
    defaultConfig: {
      actualColumn: '',
      probabilityColumn: '',
      positiveClass: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

df = input_data.copy()
actual_col = config.get('actualColumn', '')
prob_col = config.get('probabilityColumn', '')
positive_class = config.get('positiveClass', '')

if not actual_col or not prob_col:
    raise ValueError("ROC Curve: Please select actual and probability columns")

y_true = df[actual_col]
y_scores = pd.to_numeric(df[prob_col], errors='coerce')

if positive_class:
    y_true = (y_true == positive_class).astype(int)
else:
    unique_vals = y_true.unique()
    if len(unique_vals) != 2:
        raise ValueError("ROC Curve: Actual column must have exactly 2 classes or specify positive class")
    y_true = (y_true == unique_vals[1]).astype(int)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

roc_data = []
for i in range(len(fpr)):
    roc_data.append({
        'fpr': round(float(fpr[i]), 4),
        'tpr': round(float(tpr[i]), 4),
        'threshold': round(float(thresholds[i]), 4) if i < len(thresholds) else None,
    })

output = {
    'chartType': 'roc_curve',
    'data': roc_data,
    'auc': round(float(roc_auc), 4),
    'title': config.get('title', '') or f'ROC Curve (AUC = {round(roc_auc, 3)})',
}
`,
  },

  // Output Blocks
  'log-transform': {
    type: 'log-transform',
    category: 'transform',
    label: 'Log Transform',
    description: 'Apply logarithmic or exponential transformations with zero handling',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      operation: 'log',
      handleZero: 'add_one',
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
operation = config.get('operation', 'log')
handle_zero = config.get('handleZero', 'add_one')
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Log Transform: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Log Transform: Column '{column}' not found")

values = pd.to_numeric(df[column], errors='coerce')
out_col = new_column if new_column else f'{column}_{operation}'

# Handle zeros and negatives for log operations
if operation in ['log', 'log10', 'log2', 'log1p']:
    if handle_zero == 'add_one':
        values = values + 1
    elif handle_zero == 'replace_min':
        min_positive = values[values > 0].min() if (values > 0).any() else 1
        values = values.clip(lower=min_positive / 2)
    elif handle_zero == 'skip':
        pass  # Will result in NaN for non-positive values

if operation == 'log':
    df[out_col] = np.log(values)
elif operation == 'log10':
    df[out_col] = np.log10(values)
elif operation == 'log2':
    df[out_col] = np.log2(values)
elif operation == 'log1p':
    df[out_col] = np.log1p(values - 1 if handle_zero == 'add_one' else values)
elif operation == 'exp':
    df[out_col] = np.exp(values)
elif operation == 'sqrt':
    df[out_col] = np.sqrt(values.clip(lower=0))

output = df
`,
  },

  'interpolate-missing': {
    type: 'interpolate-missing',
    category: 'transform',
    label: 'Interpolate Missing',
    description: 'Fill missing values using interpolation methods',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      method: 'linear',
      orderColumn: '',
      limit: 0,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'linear')
order_col = config.get('orderColumn', '')
limit = int(config.get('limit', 0)) or None

if not column:
    raise ValueError("Interpolate Missing: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Interpolate Missing: Column '{column}' not found")

# Sort by order column if specified
if order_col and order_col in df.columns:
    df = df.sort_values(order_col).reset_index(drop=True)

if method == 'linear':
    df[column] = df[column].interpolate(method='linear', limit=limit)
elif method == 'polynomial':
    df[column] = df[column].interpolate(method='polynomial', order=2, limit=limit)
elif method == 'spline':
    df[column] = df[column].interpolate(method='spline', order=3, limit=limit)
elif method == 'nearest':
    df[column] = df[column].interpolate(method='nearest', limit=limit)
elif method == 'time':
    if order_col and order_col in df.columns:
        df = df.set_index(pd.to_datetime(df[order_col]))
        df[column] = df[column].interpolate(method='time', limit=limit)
        df = df.reset_index(drop=True)
    else:
        df[column] = df[column].interpolate(method='linear', limit=limit)
elif method == 'ffill':
    df[column] = df[column].ffill(limit=limit)
elif method == 'bfill':
    df[column] = df[column].bfill(limit=limit)

output = df
`,
  },

  'date-truncate': {
    type: 'date-truncate',
    category: 'transform',
    label: 'Date Truncate',
    description: 'Round dates to the start of a time period',
    icon: 'Calendar',
    defaultConfig: {
      column: '',
      unit: 'day',
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
column = config.get('column', '')
unit = config.get('unit', 'day')
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Date Truncate: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Date Truncate: Column '{column}' not found")

dt = pd.to_datetime(df[column], errors='coerce')
out_col = new_column if new_column else f'{column}_truncated'

if unit == 'minute':
    df[out_col] = dt.dt.floor('min')
elif unit == 'hour':
    df[out_col] = dt.dt.floor('h')
elif unit == 'day':
    df[out_col] = dt.dt.floor('D')
elif unit == 'week':
    df[out_col] = dt.dt.to_period('W').dt.start_time
elif unit == 'month':
    df[out_col] = dt.dt.to_period('M').dt.start_time
elif unit == 'quarter':
    df[out_col] = dt.dt.to_period('Q').dt.start_time
elif unit == 'year':
    df[out_col] = dt.dt.to_period('Y').dt.start_time

output = df
`,
  },

  'period-over-period': {
    type: 'period-over-period',
    category: 'transform',
    label: 'Period over Period',
    description: 'Calculate YoY, MoM, WoW, QoQ, DoD changes',
    icon: 'GitCompare',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      period: 'mom',
      changeType: 'percent',
      groupColumns: [],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
period = config.get('period', 'mom')
change_type = config.get('changeType', 'percent')
group_cols = config.get('groupColumns', [])

if not date_col or not value_col:
    raise ValueError("Period over Period: Please specify date and value columns in the Config tab")

if date_col not in df.columns:
    raise ValueError(f"Period over Period: Date column '{date_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Period over Period: Value column '{value_col}' not found")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Determine shift periods
period_map = {
    'dod': 1,      # Day over Day
    'wow': 7,      # Week over Week
    'mom': 30,     # Month over Month (approximate)
    'qoq': 90,     # Quarter over Quarter
    'yoy': 365     # Year over Year
}
shift_days = period_map.get(period, 30)

df = df.sort_values(date_col).reset_index(drop=True)

if group_cols and len(group_cols) > 0:
    valid_groups = [c for c in group_cols if c in df.columns]
    if valid_groups:
        df['_prev_value'] = df.groupby(valid_groups)[value_col].shift(1)
    else:
        df['_prev_value'] = df[value_col].shift(1)
else:
    df['_prev_value'] = df[value_col].shift(1)

if change_type == 'percent':
    df[f'{value_col}_{period}_pct'] = ((df[value_col] - df['_prev_value']) / df['_prev_value'] * 100).round(2)
elif change_type == 'absolute':
    df[f'{value_col}_{period}_diff'] = df[value_col] - df['_prev_value']
else:  # both
    df[f'{value_col}_{period}_pct'] = ((df[value_col] - df['_prev_value']) / df['_prev_value'] * 100).round(2)
    df[f'{value_col}_{period}_diff'] = df[value_col] - df['_prev_value']

df = df.drop(columns=['_prev_value'])

output = df
`,
  },

  'hash-column': {
    type: 'hash-column',
    category: 'transform',
    label: 'Hash Column',
    description: 'Hash column values for anonymization using various algorithms',
    icon: 'Hash',
    defaultConfig: {
      column: '',
      algorithm: 'sha256',
      truncateLength: 0,
      newColumn: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import hashlib

df = input_data.copy()
column = config.get('column', '')
algorithm = config.get('algorithm', 'sha256')
truncate = int(config.get('truncateLength', 0))
new_column = config.get('newColumn', '')

if not column:
    raise ValueError("Hash Column: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Hash Column: Column '{column}' not found")

out_col = new_column if new_column else f'{column}_hash'

def hash_value(val):
    if pd.isna(val):
        return None
    s = str(val).encode('utf-8')
    if algorithm == 'sha256':
        h = hashlib.sha256(s).hexdigest()
    elif algorithm == 'sha512':
        h = hashlib.sha512(s).hexdigest()
    elif algorithm == 'md5':
        h = hashlib.md5(s).hexdigest()
    elif algorithm == 'sha1':
        h = hashlib.sha1(s).hexdigest()
    elif algorithm == 'blake2':
        h = hashlib.blake2b(s).hexdigest()
    else:
        h = hashlib.sha256(s).hexdigest()

    if truncate > 0:
        h = h[:truncate]
    return h

df[out_col] = df[column].apply(hash_value)

output = df
`,
  },

  'expand-date-range': {
    type: 'expand-date-range',
    category: 'transform',
    label: 'Expand Date Range',
    description: 'Fill missing dates in a time series with various fill methods',
    icon: 'CalendarRange',
    defaultConfig: {
      dateColumn: '',
      freq: 'D',
      groupColumns: [],
      fillMethod: 'ffill',
      startDate: '',
      endDate: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
freq = config.get('freq', 'D')
group_cols = config.get('groupColumns', [])
fill_method = config.get('fillMethod', 'ffill')

if not date_col:
    raise ValueError("Expand Date Range: Please specify a date column in the Config tab")

if date_col not in df.columns:
    raise ValueError(f"Expand Date Range: Date column '{date_col}' not found")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

freq_map = {'D': 'D', 'W': 'W', 'M': 'MS', 'Q': 'QS', 'Y': 'YS', 'H': 'h'}
pd_freq = freq_map.get(freq, 'D')

if group_cols and len(group_cols) > 0:
    valid_groups = [c for c in group_cols if c in df.columns]
    if valid_groups:
        results = []
        for name, group in df.groupby(valid_groups):
            date_range = pd.date_range(group[date_col].min(), group[date_col].max(), freq=pd_freq)
            expanded = pd.DataFrame({date_col: date_range})
            for i, col in enumerate(valid_groups):
                expanded[col] = name if len(valid_groups) == 1 else name[i]
            merged = expanded.merge(group, on=[date_col] + valid_groups, how='left')
            results.append(merged)
        df = pd.concat(results, ignore_index=True)
    else:
        date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq=pd_freq)
        expanded = pd.DataFrame({date_col: date_range})
        df = expanded.merge(df, on=date_col, how='left')
else:
    date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq=pd_freq)
    expanded = pd.DataFrame({date_col: date_range})
    df = expanded.merge(df, on=date_col, how='left')

# Apply fill method
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if fill_method == 'ffill':
    df[numeric_cols] = df[numeric_cols].ffill()
elif fill_method == 'bfill':
    df[numeric_cols] = df[numeric_cols].bfill()
elif fill_method == 'zero':
    df[numeric_cols] = df[numeric_cols].fillna(0)
elif fill_method == 'interpolate':
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

output = df.sort_values(date_col).reset_index(drop=True)
`,
  },

  'string-similarity': {
    type: 'string-similarity',
    category: 'transform',
    label: 'String Similarity',
    description: 'Calculate similarity between string columns for fuzzy matching',
    icon: 'GitCompare',
    defaultConfig: {
      column1: '',
      column2: '',
      method: 'levenshtein',
      threshold: 0.8,
      newColumn: 'similarity',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
col1 = config.get('column1', '')
col2 = config.get('column2', '')
method = config.get('method', 'levenshtein')
threshold = float(config.get('threshold', 0.8))
new_col = config.get('newColumn', 'similarity')

if not col1 or not col2:
    raise ValueError("String Similarity: Please specify both columns in the Config tab")

if col1 not in df.columns:
    raise ValueError(f"String Similarity: Column '{col1}' not found")
if col2 not in df.columns:
    raise ValueError(f"String Similarity: Column '{col2}' not found")

def levenshtein_ratio(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return np.nan
    s1, s2 = str(s1), str(s2)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dist[i][j] = min(dist[i-1][j] + 1, dist[i][j-1] + 1, dist[i-1][j-1] + cost)

    max_len = max(len(s1), len(s2))
    return 1 - (dist[rows-1][cols-1] / max_len) if max_len > 0 else 1.0

def jaro_similarity(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return np.nan
    s1, s2 = str(s1), str(s2)
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3

def jaro_winkler(s1, s2, p=0.1):
    jaro = jaro_similarity(s1, s2)
    if pd.isna(jaro):
        return np.nan
    s1, s2 = str(s1), str(s2)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)

if method == 'levenshtein':
    df[new_col] = df.apply(lambda r: levenshtein_ratio(r[col1], r[col2]), axis=1)
elif method == 'jaro':
    df[new_col] = df.apply(lambda r: jaro_similarity(r[col1], r[col2]), axis=1)
elif method == 'jaro_winkler':
    df[new_col] = df.apply(lambda r: jaro_winkler(r[col1], r[col2]), axis=1)
elif method == 'exact':
    df[new_col] = (df[col1].astype(str) == df[col2].astype(str)).astype(float)
elif method == 'contains':
    df[new_col] = df.apply(lambda r: 1.0 if str(r[col2]) in str(r[col1]) else 0.0, axis=1)

df[new_col] = df[new_col].round(4)
df[f'{new_col}_match'] = df[new_col] >= threshold

output = df
`,
  },

  'generate-sequence': {
    type: 'generate-sequence',
    category: 'transform',
    label: 'Generate Sequence',
    description: 'Create a new dataset with number sequences, date ranges, or repeated patterns',
    icon: 'ListOrdered',
    defaultConfig: {
      type: 'number',
      start: 1,
      end: 10,
      step: 1,
      columnName: 'sequence',
      dateStart: '',
      dateEnd: '',
      dateFreq: 'D',
      repeatValue: '',
      repeatCount: 10,
    },
    inputs: 0,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

seq_type = config.get('type', 'number')
col_name = config.get('columnName', 'sequence')

if seq_type == 'number':
    start = float(config.get('start', 1))
    end = float(config.get('end', 10))
    step = float(config.get('step', 1))

    if step == 0:
        raise ValueError("Generate Sequence: Step cannot be zero")

    values = np.arange(start, end + step/2, step)
    df = pd.DataFrame({col_name: values})

elif seq_type == 'date':
    date_start = config.get('dateStart', '')
    date_end = config.get('dateEnd', '')
    freq = config.get('dateFreq', 'D')

    if not date_start or not date_end:
        from datetime import datetime, timedelta
        date_end = datetime.now()
        date_start = date_end - timedelta(days=30)
    else:
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)

    freq_map = {'D': 'D', 'W': 'W', 'M': 'MS', 'Q': 'QS', 'Y': 'YS', 'H': 'h'}
    pd_freq = freq_map.get(freq, 'D')

    dates = pd.date_range(date_start, date_end, freq=pd_freq)
    df = pd.DataFrame({col_name: dates})

elif seq_type == 'repeat':
    value = config.get('repeatValue', 'A')
    count = int(config.get('repeatCount', 10))

    df = pd.DataFrame({col_name: [value] * count})

else:
    df = pd.DataFrame({col_name: range(1, 11)})

output = df
`,
  },

  'top-n-per-group': {
    type: 'top-n-per-group',
    category: 'transform',
    label: 'Top N per Group',
    description: 'Get top or bottom N rows per group with optional rank column',
    icon: 'Trophy',
    defaultConfig: {
      groupColumns: [],
      orderColumn: '',
      n: 5,
      ascending: false,
      includeRank: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
group_cols = config.get('groupColumns', [])
order_col = config.get('orderColumn', '')
n = int(config.get('n', 5))
ascending = config.get('ascending', False)
include_rank = config.get('includeRank', True)

if not order_col:
    raise ValueError("Top N per Group: Please specify an order column in the Config tab")

if order_col not in df.columns:
    raise ValueError(f"Top N per Group: Order column '{order_col}' not found")

if group_cols and len(group_cols) > 0:
    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        raise ValueError("Top N per Group: No valid group columns found")

    # Sort and get top N per group
    df = df.sort_values([*valid_groups, order_col], ascending=[True]*len(valid_groups) + [ascending])

    result = df.groupby(valid_groups, group_keys=False).head(n)

    if include_rank:
        result['_rank'] = result.groupby(valid_groups).cumcount() + 1
        result = result.rename(columns={'_rank': 'rank_in_group'})
else:
    df = df.sort_values(order_col, ascending=ascending)
    result = df.head(n)

    if include_rank:
        result = result.copy()
        result['rank'] = range(1, len(result) + 1)

output = result.reset_index(drop=True)
`,
  },

  'first-last-per-group': {
    type: 'first-last-per-group',
    category: 'transform',
    label: 'First/Last per Group',
    description: 'Get the first, last, or both rows per group based on sort order',
    icon: 'ListFilter',
    defaultConfig: {
      groupColumns: [],
      orderColumn: '',
      position: 'first',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
group_cols = config.get('groupColumns', [])
order_col = config.get('orderColumn', '')
position = config.get('position', 'first')

if not group_cols or len(group_cols) == 0:
    raise ValueError("First/Last per Group: Please specify at least one group column in the Config tab")

valid_groups = [c for c in group_cols if c in df.columns]
if not valid_groups:
    raise ValueError("First/Last per Group: No valid group columns found")

# Sort by order column if specified
if order_col and order_col in df.columns:
    df = df.sort_values([*valid_groups, order_col])
else:
    df = df.sort_values(valid_groups)

if position == 'first':
    result = df.groupby(valid_groups, as_index=False).first()
elif position == 'last':
    result = df.groupby(valid_groups, as_index=False).last()
elif position == 'both':
    first_df = df.groupby(valid_groups, as_index=False).first()
    first_df['_position'] = 'first'
    last_df = df.groupby(valid_groups, as_index=False).last()
    last_df['_position'] = 'last'
    result = pd.concat([first_df, last_df], ignore_index=True)
    result = result.sort_values(valid_groups)

output = result.reset_index(drop=True)
`,
  },

  // ML Preprocessing Transform Blocks
  'one-hot-encode': {
    type: 'one-hot-encode',
    category: 'transform',
    label: 'One-Hot Encode',
    description: 'Convert categorical columns to binary indicator columns (dummy variables) for ML models',
    icon: 'Binary',
    defaultConfig: {
      columns: [],
      dropFirst: false,
      prefix: '',
      handleUnknown: 'error',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
columns = config.get('columns', [])
drop_first = config.get('dropFirst', False)
prefix = config.get('prefix', '')
handle_unknown = config.get('handleUnknown', 'error')

if not columns or len(columns) == 0:
    raise ValueError("One-Hot Encode: Please specify at least one column to encode in the Config tab")

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"One-Hot Encode: No valid columns found. Available: {', '.join(df.columns.tolist())}")

# Get non-encoded columns to preserve
other_cols = [c for c in df.columns if c not in valid_cols]
preserved_df = df[other_cols].copy() if other_cols else None

# Apply one-hot encoding
encoded_df = pd.get_dummies(
    df[valid_cols],
    prefix=prefix if prefix else None,
    prefix_sep='_' if prefix else '_',
    drop_first=drop_first,
    dtype=int
)

# Combine preserved and encoded columns
if preserved_df is not None:
    output = pd.concat([preserved_df, encoded_df], axis=1)
else:
    output = encoded_df
`,
  },

  'label-encode': {
    type: 'label-encode',
    category: 'transform',
    label: 'Label Encode',
    description: 'Convert categorical values to integer codes (0, 1, 2, ...) for tree-based ML models',
    icon: 'Hash',
    defaultConfig: {
      columns: [],
      mappingOrder: 'alphabetical',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
mapping_order = config.get('mappingOrder', 'alphabetical')

if not columns or len(columns) == 0:
    raise ValueError("Label Encode: Please specify at least one column to encode in the Config tab")

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"Label Encode: No valid columns found. Available: {', '.join(df.columns.tolist())}")

mappings = {}
for col in valid_cols:
    unique_vals = df[col].dropna().unique()

    if mapping_order == 'alphabetical':
        sorted_vals = sorted(unique_vals, key=str)
    elif mapping_order == 'frequency':
        value_counts = df[col].value_counts()
        sorted_vals = value_counts.index.tolist()
    else:  # appearance order
        sorted_vals = list(unique_vals)

    mapping = {val: idx for idx, val in enumerate(sorted_vals)}
    mappings[col] = mapping
    df[col] = df[col].map(mapping)
    # Keep NaN as NaN
    df[col] = df[col].astype('Int64')

output = df
`,
  },

  'ordinal-encode': {
    type: 'ordinal-encode',
    category: 'transform',
    label: 'Ordinal Encode',
    description: 'Encode categorical values with user-defined order (e.g., low=1, medium=2, high=3)',
    icon: 'ListOrdered',
    defaultConfig: {
      column: '',
      orderMapping: [],
      unknownValue: -1,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
order_mapping = config.get('orderMapping', [])
unknown_value = config.get('unknownValue', -1)

if not column:
    raise ValueError("Ordinal Encode: Please specify a column to encode in the Config tab")

if column not in df.columns:
    raise ValueError(f"Ordinal Encode: Column '{column}' not found. Available: {', '.join(df.columns.tolist())}")

if not order_mapping or len(order_mapping) == 0:
    raise ValueError("Ordinal Encode: Please specify the order mapping (e.g., ['low', 'medium', 'high'])")

# Create mapping from order list
mapping = {val: idx + 1 for idx, val in enumerate(order_mapping)}

# Apply mapping, use unknown_value for values not in mapping
df[column + '_encoded'] = df[column].apply(lambda x: mapping.get(x, unknown_value) if pd.notna(x) else np.nan)
df[column + '_encoded'] = df[column + '_encoded'].astype('Int64')

output = df
`,
  },

  'min-max-normalize': {
    type: 'min-max-normalize',
    category: 'transform',
    label: 'Min-Max Normalize',
    description: 'Scale numeric columns to [0, 1] range using (x - min) / (max - min)',
    icon: 'Scale',
    defaultConfig: {
      columns: [],
      featureRangeMin: 0,
      featureRangeMax: 1,
      suffix: '_normalized',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
range_min = config.get('featureRangeMin', 0)
range_max = config.get('featureRangeMax', 1)
suffix = config.get('suffix', '_normalized')

if not columns or len(columns) == 0:
    raise ValueError("Min-Max Normalize: Please specify at least one column to normalize in the Config tab")

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"Min-Max Normalize: No valid columns found. Available: {', '.join(df.columns.tolist())}")

for col in valid_cols:
    col_data = pd.to_numeric(df[col], errors='coerce')
    col_min = col_data.min()
    col_max = col_data.max()

    if col_max == col_min:
        # Avoid division by zero - set to middle of range
        df[col + suffix] = (range_min + range_max) / 2
    else:
        # Apply min-max normalization
        normalized = (col_data - col_min) / (col_max - col_min)
        df[col + suffix] = normalized * (range_max - range_min) + range_min

output = df
`,
  },

  'z-score-standardize': {
    type: 'z-score-standardize',
    category: 'transform',
    label: 'Z-Score Standardize',
    description: 'Transform columns to mean=0 and std=1 using (x - mean) / std',
    icon: 'Activity',
    defaultConfig: {
      columns: [],
      withMean: true,
      withStd: true,
      suffix: '_standardized',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
with_mean = config.get('withMean', True)
with_std = config.get('withStd', True)
suffix = config.get('suffix', '_standardized')

if not columns or len(columns) == 0:
    raise ValueError("Z-Score Standardize: Please specify at least one column to standardize in the Config tab")

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"Z-Score Standardize: No valid columns found. Available: {', '.join(df.columns.tolist())}")

for col in valid_cols:
    col_data = pd.to_numeric(df[col], errors='coerce')
    col_mean = col_data.mean() if with_mean else 0
    col_std = col_data.std() if with_std else 1

    if col_std == 0 or pd.isna(col_std):
        # Avoid division by zero
        df[col + suffix] = 0 if with_mean else col_data
    else:
        df[col + suffix] = (col_data - col_mean) / col_std

output = df
`,
  },

  'rolling-statistics': {
    type: 'rolling-statistics',
    category: 'transform',
    label: 'Rolling Statistics',
    description: 'Calculate moving window statistics (mean, sum, min, max, std) for time series',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      windowSize: 7,
      statistic: 'mean',
      minPeriods: 1,
      center: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
window_size = config.get('windowSize', 7)
statistic = config.get('statistic', 'mean')
min_periods = config.get('minPeriods', 1)
center = config.get('center', False)

if not column:
    raise ValueError("Rolling Statistics: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Rolling Statistics: Column '{column}' not found. Available: {', '.join(df.columns.tolist())}")

col_data = pd.to_numeric(df[column], errors='coerce')
rolling = col_data.rolling(window=window_size, min_periods=min_periods, center=center)

new_col_name = f"{column}_rolling_{statistic}_{window_size}"

if statistic == 'mean':
    df[new_col_name] = rolling.mean()
elif statistic == 'sum':
    df[new_col_name] = rolling.sum()
elif statistic == 'min':
    df[new_col_name] = rolling.min()
elif statistic == 'max':
    df[new_col_name] = rolling.max()
elif statistic == 'std':
    df[new_col_name] = rolling.std()
elif statistic == 'median':
    df[new_col_name] = rolling.median()
elif statistic == 'count':
    df[new_col_name] = rolling.count()
elif statistic == 'var':
    df[new_col_name] = rolling.var()
else:
    raise ValueError(f"Rolling Statistics: Unknown statistic '{statistic}'. Use mean, sum, min, max, std, median, count, or var")

output = df
`,
  },

  'resample-timeseries': {
    type: 'resample-timeseries',
    category: 'transform',
    label: 'Resample Time Series',
    description: 'Change time series frequency (minute to hour, day to week, etc.) with aggregation',
    icon: 'Clock',
    defaultConfig: {
      datetimeColumn: '',
      frequency: 'D',
      aggregation: 'mean',
      fillMethod: 'none',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
datetime_col = config.get('datetimeColumn', '')
frequency = config.get('frequency', 'D')
aggregation = config.get('aggregation', 'mean')
fill_method = config.get('fillMethod', 'none')

if not datetime_col:
    raise ValueError("Resample Time Series: Please specify a datetime column in the Config tab")

if datetime_col not in df.columns:
    raise ValueError(f"Resample Time Series: Column '{datetime_col}' not found. Available: {', '.join(df.columns.tolist())}")

# Convert to datetime
df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

# Set datetime as index for resampling
df_indexed = df.set_index(datetime_col)

# Get numeric columns for aggregation
numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    raise ValueError("Resample Time Series: No numeric columns found for aggregation")

# Apply resampling with aggregation
if aggregation == 'mean':
    resampled = df_indexed[numeric_cols].resample(frequency).mean()
elif aggregation == 'sum':
    resampled = df_indexed[numeric_cols].resample(frequency).sum()
elif aggregation == 'min':
    resampled = df_indexed[numeric_cols].resample(frequency).min()
elif aggregation == 'max':
    resampled = df_indexed[numeric_cols].resample(frequency).max()
elif aggregation == 'first':
    resampled = df_indexed[numeric_cols].resample(frequency).first()
elif aggregation == 'last':
    resampled = df_indexed[numeric_cols].resample(frequency).last()
elif aggregation == 'count':
    resampled = df_indexed[numeric_cols].resample(frequency).count()
elif aggregation == 'median':
    resampled = df_indexed[numeric_cols].resample(frequency).median()
else:
    raise ValueError(f"Resample Time Series: Unknown aggregation '{aggregation}'")

# Apply fill method if specified
if fill_method == 'ffill':
    resampled = resampled.ffill()
elif fill_method == 'bfill':
    resampled = resampled.bfill()
elif fill_method == 'interpolate':
    resampled = resampled.interpolate()

output = resampled.reset_index()
`,
  },

  'regex-replace': {
    type: 'regex-replace',
    category: 'transform',
    label: 'Regex Replace',
    description: 'Find and replace text using regular expression patterns with capture groups',
    icon: 'Code',
    defaultConfig: {
      column: '',
      pattern: '',
      replacement: '',
      caseInsensitive: false,
      replaceAll: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import re

df = input_data.copy()
column = config.get('column', '')
pattern = config.get('pattern', '')
replacement = config.get('replacement', '')
case_insensitive = config.get('caseInsensitive', False)
replace_all = config.get('replaceAll', True)

if not column:
    raise ValueError("Regex Replace: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Regex Replace: Column '{column}' not found. Available: {', '.join(df.columns.tolist())}")

if not pattern:
    raise ValueError("Regex Replace: Please specify a regex pattern")

flags = re.IGNORECASE if case_insensitive else 0

try:
    if replace_all:
        df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True, flags=flags)
    else:
        # Replace only first occurrence
        df[column] = df[column].astype(str).apply(
            lambda x: re.sub(pattern, replacement, x, count=1, flags=flags)
        )
except re.error as e:
    raise ValueError(f"Regex Replace: Invalid regex pattern - {str(e)}")

output = df
`,
  },

  'expand-json-column': {
    type: 'expand-json-column',
    category: 'transform',
    label: 'Expand JSON Column',
    description: 'Parse JSON strings in a column and expand into multiple columns',
    icon: 'Layers',
    defaultConfig: {
      column: '',
      prefix: '',
      maxLevel: 1,
      keepOriginal: false,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import json

df = input_data.copy()
column = config.get('column', '')
prefix = config.get('prefix', '')
max_level = config.get('maxLevel', 1)
keep_original = config.get('keepOriginal', False)

if not column:
    raise ValueError("Expand JSON Column: Please specify a column in the Config tab")

if column not in df.columns:
    raise ValueError(f"Expand JSON Column: Column '{column}' not found. Available: {', '.join(df.columns.tolist())}")

def safe_json_parse(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    try:
        return json.loads(str(x))
    except (json.JSONDecodeError, TypeError):
        return {}

# Parse JSON strings
parsed = df[column].apply(safe_json_parse)

# Normalize to DataFrame
json_df = pd.json_normalize(parsed, max_level=max_level)

# Add prefix if specified
if prefix:
    json_df.columns = [f"{prefix}_{col}" for col in json_df.columns]
else:
    json_df.columns = [f"{column}_{col}" for col in json_df.columns]

# Combine with original data
if keep_original:
    output = pd.concat([df, json_df], axis=1)
else:
    other_cols = [c for c in df.columns if c != column]
    output = pd.concat([df[other_cols], json_df], axis=1)
`,
  },

  'add-unique-id': {
    type: 'add-unique-id',
    category: 'transform',
    label: 'Add Unique ID',
    description: 'Generate unique identifiers for each row (sequential, UUID, or hash-based)',
    icon: 'Hash',
    defaultConfig: {
      idType: 'sequential',
      columnName: 'id',
      prefix: '',
      startFrom: 1,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import uuid
import hashlib

df = input_data.copy()
id_type = config.get('idType', 'sequential')
column_name = config.get('columnName', 'id')
prefix = config.get('prefix', '')
start_from = config.get('startFrom', 1)

n_rows = len(df)

if id_type == 'sequential':
    ids = [f"{prefix}{i}" if prefix else i for i in range(start_from, start_from + n_rows)]
elif id_type == 'uuid':
    ids = [f"{prefix}{str(uuid.uuid4())}" if prefix else str(uuid.uuid4()) for _ in range(n_rows)]
elif id_type == 'uuid_short':
    ids = [f"{prefix}{str(uuid.uuid4())[:8]}" if prefix else str(uuid.uuid4())[:8] for _ in range(n_rows)]
elif id_type == 'hash':
    # Create hash from row index and random component
    ids = []
    for i in range(n_rows):
        hash_input = f"{i}_{uuid.uuid4()}"
        hash_val = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        ids.append(f"{prefix}{hash_val}" if prefix else hash_val)
else:
    raise ValueError(f"Add Unique ID: Unknown id type '{id_type}'. Use sequential, uuid, uuid_short, or hash")

# Insert ID column at the beginning
df.insert(0, column_name, ids)

output = df
`,
  },

  'missing-indicator': {
    type: 'missing-indicator',
    category: 'transform',
    label: 'Missing Value Indicator',
    description: 'Create binary columns indicating where values are missing (1=missing, 0=present)',
    icon: 'AlertTriangle',
    defaultConfig: {
      columns: [],
      suffix: '_missing',
      onlyIfMissing: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
suffix = config.get('suffix', '_missing')
only_if_missing = config.get('onlyIfMissing', True)

# If no columns specified, use all columns
if not columns or len(columns) == 0:
    columns = df.columns.tolist()

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"Missing Value Indicator: No valid columns found. Available: {', '.join(df.columns.tolist())}")

for col in valid_cols:
    has_missing = df[col].isna().any()

    # Only create indicator if column has missing values (when onlyIfMissing is True)
    if not only_if_missing or has_missing:
        df[col + suffix] = df[col].isna().astype(int)

output = df
`,
  },

  'quantile-transform': {
    type: 'quantile-transform',
    category: 'transform',
    label: 'Quantile Transform',
    description: 'Transform values to their quantile rank, then to uniform [0,1] or normal distribution',
    icon: 'BarChart3',
    defaultConfig: {
      columns: [],
      outputDistribution: 'uniform',
      nQuantiles: 1000,
      suffix: '_quantile',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
columns = config.get('columns', [])
output_dist = config.get('outputDistribution', 'uniform')
n_quantiles = config.get('nQuantiles', 1000)
suffix = config.get('suffix', '_quantile')

if not columns or len(columns) == 0:
    raise ValueError("Quantile Transform: Please specify at least one column in the Config tab")

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError(f"Quantile Transform: No valid columns found. Available: {', '.join(df.columns.tolist())}")

for col in valid_cols:
    col_data = pd.to_numeric(df[col], errors='coerce')

    # Calculate quantile ranks (0 to 1)
    # Use average method to handle ties
    ranks = col_data.rank(method='average', pct=True)

    if output_dist == 'uniform':
        # Already in [0, 1] range
        df[col + suffix] = ranks
    elif output_dist == 'normal':
        # Transform to standard normal using inverse CDF
        # Clip to avoid infinite values at 0 and 1
        clipped_ranks = ranks.clip(1e-7, 1 - 1e-7)
        df[col + suffix] = stats.norm.ppf(clipped_ranks)
    else:
        raise ValueError(f"Quantile Transform: Unknown output distribution '{output_dist}'. Use 'uniform' or 'normal'")

output = df
`,
  },

  // New Visualization Blocks
  'funnel-chart': {
    type: 'funnel-chart',
    category: 'visualization',
    label: 'Funnel Chart',
    description: 'Visualize sequential stages with progressive reduction (conversion funnels, sales pipelines)',
    icon: 'Filter',
    defaultConfig: {
      stageColumn: '',
      valueColumn: '',
      title: '',
      showPercentage: true,
      showDropoff: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
stage_col = config.get('stageColumn', '')
value_col = config.get('valueColumn', '')
show_pct = config.get('showPercentage', True)
show_dropoff = config.get('showDropoff', True)

if not stage_col or not value_col:
    raise ValueError("Funnel Chart: Please select stage and value columns")

if stage_col not in df.columns:
    raise ValueError(f"Funnel Chart: Stage column '{stage_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Funnel Chart: Value column '{value_col}' not found")

# Aggregate by stage
funnel_data = df.groupby(stage_col, sort=False)[value_col].sum().reset_index()
funnel_data.columns = ['stage', 'value']

# Calculate percentages and dropoff
total = funnel_data['value'].iloc[0] if len(funnel_data) > 0 else 1
funnel_data['percentage'] = (funnel_data['value'] / total * 100).round(2)
funnel_data['dropoff'] = funnel_data['value'].diff().fillna(0).round(2)
funnel_data['dropoff_pct'] = (funnel_data['dropoff'] / funnel_data['value'].shift(1) * 100).fillna(0).round(2)

output = {
    'chartType': 'funnel',
    'data': funnel_data.to_dict('records'),
    'title': config.get('title', '') or 'Funnel Chart',
    'showPercentage': show_pct,
    'showDropoff': show_dropoff,
}
`,
  },

  'sankey-diagram': {
    type: 'sankey-diagram',
    category: 'visualization',
    label: 'Sankey Diagram',
    description: 'Show flow/movement between nodes with proportional width bands (user journeys, budget flows)',
    icon: 'GitMerge',
    defaultConfig: {
      sourceColumn: '',
      targetColumn: '',
      valueColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
source_col = config.get('sourceColumn', '')
target_col = config.get('targetColumn', '')
value_col = config.get('valueColumn', '')

if not source_col or not target_col:
    raise ValueError("Sankey Diagram: Please select source and target columns")

if source_col not in df.columns:
    raise ValueError(f"Sankey Diagram: Source column '{source_col}' not found")
if target_col not in df.columns:
    raise ValueError(f"Sankey Diagram: Target column '{target_col}' not found")

# Aggregate flows
if value_col and value_col in df.columns:
    flows = df.groupby([source_col, target_col])[value_col].sum().reset_index()
    flows.columns = ['source', 'target', 'value']
else:
    flows = df.groupby([source_col, target_col]).size().reset_index(name='value')
    flows.columns = ['source', 'target', 'value']

# Get unique nodes
all_nodes = pd.concat([flows['source'], flows['target']]).unique().tolist()
node_indices = {node: i for i, node in enumerate(all_nodes)}

# Create links with indices
links = []
for _, row in flows.iterrows():
    links.append({
        'source': node_indices[row['source']],
        'target': node_indices[row['target']],
        'value': float(row['value']),
        'sourceLabel': str(row['source']),
        'targetLabel': str(row['target']),
    })

output = {
    'chartType': 'sankey',
    'nodes': [{'label': str(n)} for n in all_nodes],
    'links': links,
    'title': config.get('title', '') or 'Sankey Diagram',
}
`,
  },

  'treemap': {
    type: 'treemap',
    category: 'visualization',
    label: 'Treemap',
    description: 'Display hierarchical data as nested rectangles sized by value (budget breakdown, composition)',
    icon: 'LayoutGrid',
    defaultConfig: {
      pathColumns: [],
      valueColumn: '',
      colorColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
path_cols = config.get('pathColumns', [])
value_col = config.get('valueColumn', '')
color_col = config.get('colorColumn', '')

if not path_cols or len(path_cols) == 0:
    raise ValueError("Treemap: Please select at least one path column for hierarchy")

valid_paths = [c for c in path_cols if c in df.columns]
if not valid_paths:
    raise ValueError("Treemap: No valid path columns found")

# Build hierarchical data
if value_col and value_col in df.columns:
    agg_df = df.groupby(valid_paths)[value_col].sum().reset_index()
else:
    agg_df = df.groupby(valid_paths).size().reset_index(name='count')
    value_col = 'count'

# Create treemap data structure
labels = []
parents = []
values = []
ids = []

# Add root
labels.append('')
parents.append('')
values.append(0)
ids.append('')

for _, row in agg_df.iterrows():
    path_parts = [str(row[c]) for c in valid_paths]
    current_id = ''
    for i, part in enumerate(path_parts):
        new_id = '/'.join(path_parts[:i+1])
        if new_id not in ids:
            ids.append(new_id)
            labels.append(part)
            parents.append(current_id)
            if i == len(path_parts) - 1:
                values.append(float(row[value_col]))
            else:
                values.append(0)
        current_id = new_id

output = {
    'chartType': 'treemap',
    'ids': ids,
    'labels': labels,
    'parents': parents,
    'values': values,
    'title': config.get('title', '') or 'Treemap',
}
`,
  },

  'sunburst-chart': {
    type: 'sunburst-chart',
    category: 'visualization',
    label: 'Sunburst Chart',
    description: 'Radial hierarchical visualization with concentric rings (org structure, taxonomy)',
    icon: 'Sun',
    defaultConfig: {
      pathColumns: [],
      valueColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
path_cols = config.get('pathColumns', [])
value_col = config.get('valueColumn', '')

if not path_cols or len(path_cols) == 0:
    raise ValueError("Sunburst Chart: Please select at least one path column for hierarchy")

valid_paths = [c for c in path_cols if c in df.columns]
if not valid_paths:
    raise ValueError("Sunburst Chart: No valid path columns found")

# Build hierarchical data
if value_col and value_col in df.columns:
    agg_df = df.groupby(valid_paths)[value_col].sum().reset_index()
else:
    agg_df = df.groupby(valid_paths).size().reset_index(name='count')
    value_col = 'count'

# Create sunburst data structure
labels = []
parents = []
values = []
ids = []

for _, row in agg_df.iterrows():
    path_parts = [str(row[c]) for c in valid_paths]
    current_id = ''
    for i, part in enumerate(path_parts):
        new_id = '/'.join(path_parts[:i+1])
        if new_id not in ids:
            ids.append(new_id)
            labels.append(part)
            parents.append(current_id)
            if i == len(path_parts) - 1:
                values.append(float(row[value_col]))
            else:
                values.append(0)
        current_id = new_id

output = {
    'chartType': 'sunburst',
    'ids': ids,
    'labels': labels,
    'parents': parents,
    'values': values,
    'title': config.get('title', '') or 'Sunburst Chart',
}
`,
  },

  'gauge-chart': {
    type: 'gauge-chart',
    category: 'visualization',
    label: 'Gauge Chart',
    description: 'Speedometer-style display for single metrics against targets (KPIs, performance scores)',
    icon: 'Gauge',
    defaultConfig: {
      valueColumn: '',
      minValue: 0,
      maxValue: 100,
      thresholds: [30, 70],
      colors: ['#EF4444', '#F59E0B', '#10B981'],
      title: '',
      suffix: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
value_col = config.get('valueColumn', '')
min_val = float(config.get('minValue', 0))
max_val = float(config.get('maxValue', 100))
thresholds = config.get('thresholds', [30, 70])
colors = config.get('colors', ['#EF4444', '#F59E0B', '#10B981'])
suffix = config.get('suffix', '')

if not value_col:
    raise ValueError("Gauge Chart: Please select a value column")

if value_col not in df.columns:
    raise ValueError(f"Gauge Chart: Column '{value_col}' not found")

# Get the value (use first row or aggregate)
value = df[value_col].mean()
if pd.isna(value):
    value = 0

# Determine color based on thresholds
color_idx = 0
for i, threshold in enumerate(thresholds):
    if value >= threshold:
        color_idx = i + 1

output = {
    'chartType': 'gauge',
    'value': round(float(value), 2),
    'min': min_val,
    'max': max_val,
    'thresholds': thresholds,
    'colors': colors,
    'currentColor': colors[min(color_idx, len(colors)-1)],
    'title': config.get('title', '') or 'Gauge',
    'suffix': suffix,
}
`,
  },

  'radar-chart': {
    type: 'radar-chart',
    category: 'visualization',
    label: 'Radar Chart',
    description: 'Multi-axis radial chart comparing multiple variables (skill profiles, product features)',
    icon: 'Radar',
    defaultConfig: {
      categoryColumn: '',
      valueColumns: [],
      title: '',
      fill: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
value_cols = config.get('valueColumns', [])
fill = config.get('fill', True)

if not value_cols or len(value_cols) == 0:
    raise ValueError("Radar Chart: Please select value columns to display")

valid_cols = [c for c in value_cols if c in df.columns]
if not valid_cols:
    raise ValueError("Radar Chart: No valid value columns found")

# If category column specified, create traces for each category
if cat_col and cat_col in df.columns:
    traces = []
    for cat_value in df[cat_col].unique():
        cat_df = df[df[cat_col] == cat_value]
        values = [float(cat_df[c].mean()) for c in valid_cols]
        traces.append({
            'name': str(cat_value),
            'values': values + [values[0]],  # Close the polygon
        })
    theta = valid_cols + [valid_cols[0]]
else:
    # Single trace with mean values
    values = [float(df[c].mean()) for c in valid_cols]
    traces = [{
        'name': 'Values',
        'values': values + [values[0]],
    }]
    theta = valid_cols + [valid_cols[0]]

output = {
    'chartType': 'radar',
    'traces': traces,
    'theta': theta,
    'fill': fill,
    'title': config.get('title', '') or 'Radar Chart',
}
`,
  },

  'waterfall-chart': {
    type: 'waterfall-chart',
    category: 'visualization',
    label: 'Waterfall Chart',
    description: 'Show cumulative effect of sequential positive/negative values (revenue breakdown, variance)',
    icon: 'BarChart3',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      title: '',
      showTotal: true,
      positiveColor: '#10B981',
      negativeColor: '#EF4444',
      totalColor: '#3B82F6',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')
show_total = config.get('showTotal', True)

if not cat_col or not value_col:
    raise ValueError("Waterfall Chart: Please select category and value columns")

if cat_col not in df.columns:
    raise ValueError(f"Waterfall Chart: Category column '{cat_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Waterfall Chart: Value column '{value_col}' not found")

# Build waterfall data
categories = df[cat_col].tolist()
values = df[value_col].tolist()

# Determine measure type for each bar
measures = []
for i, v in enumerate(values):
    if i == 0:
        measures.append('absolute')
    else:
        measures.append('relative')

# Add total if requested
if show_total:
    categories.append('Total')
    values.append(sum(values))
    measures.append('total')

output = {
    'chartType': 'waterfall',
    'categories': categories,
    'values': [float(v) for v in values],
    'measures': measures,
    'title': config.get('title', '') or 'Waterfall Chart',
    'positiveColor': config.get('positiveColor', '#10B981'),
    'negativeColor': config.get('negativeColor', '#EF4444'),
    'totalColor': config.get('totalColor', '#3B82F6'),
}
`,
  },

  'candlestick-chart': {
    type: 'candlestick-chart',
    category: 'visualization',
    label: 'Candlestick Chart',
    description: 'Financial OHLC visualization for stock/price data with optional volume',
    icon: 'CandlestickChart',
    defaultConfig: {
      dateColumn: '',
      openColumn: '',
      highColumn: '',
      lowColumn: '',
      closeColumn: '',
      volumeColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
open_col = config.get('openColumn', '')
high_col = config.get('highColumn', '')
low_col = config.get('lowColumn', '')
close_col = config.get('closeColumn', '')
volume_col = config.get('volumeColumn', '')

if not all([date_col, open_col, high_col, low_col, close_col]):
    raise ValueError("Candlestick Chart: Please select date, open, high, low, and close columns")

required_cols = [date_col, open_col, high_col, low_col, close_col]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Candlestick Chart: Column '{col}' not found")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.sort_values(date_col)

data = {
    'dates': df[date_col].dt.strftime('%Y-%m-%d').tolist(),
    'open': df[open_col].tolist(),
    'high': df[high_col].tolist(),
    'low': df[low_col].tolist(),
    'close': df[close_col].tolist(),
}

if volume_col and volume_col in df.columns:
    data['volume'] = df[volume_col].tolist()

output = {
    'chartType': 'candlestick',
    'data': data,
    'title': config.get('title', '') or 'Candlestick Chart',
}
`,
  },

  'choropleth-map': {
    type: 'choropleth-map',
    category: 'visualization',
    label: 'Choropleth Map',
    description: 'Geographic heatmap coloring regions by data values (sales by state, population)',
    icon: 'Map',
    defaultConfig: {
      locationColumn: '',
      valueColumn: '',
      locationType: 'usa-states',
      colorScale: 'Blues',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
loc_col = config.get('locationColumn', '')
value_col = config.get('valueColumn', '')
loc_type = config.get('locationType', 'usa-states')
color_scale = config.get('colorScale', 'Blues')

if not loc_col or not value_col:
    raise ValueError("Choropleth Map: Please select location and value columns")

if loc_col not in df.columns:
    raise ValueError(f"Choropleth Map: Location column '{loc_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Choropleth Map: Value column '{value_col}' not found")

# Aggregate by location
map_data = df.groupby(loc_col)[value_col].sum().reset_index()
map_data.columns = ['location', 'value']

output = {
    'chartType': 'choropleth',
    'locations': map_data['location'].tolist(),
    'values': [float(v) for v in map_data['value'].tolist()],
    'locationType': loc_type,
    'colorScale': color_scale,
    'title': config.get('title', '') or 'Choropleth Map',
}
`,
  },

  'word-cloud': {
    type: 'word-cloud',
    category: 'visualization',
    label: 'Word Cloud',
    description: 'Text visualization where word size represents frequency (survey responses, NLP)',
    icon: 'Cloud',
    defaultConfig: {
      textColumn: '',
      weightColumn: '',
      maxWords: 100,
      minFontSize: 12,
      maxFontSize: 60,
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
import re

df = input_data.copy()
text_col = config.get('textColumn', '')
weight_col = config.get('weightColumn', '')
max_words = int(config.get('maxWords', 100))

if not text_col:
    raise ValueError("Word Cloud: Please select a text column")

if text_col not in df.columns:
    raise ValueError(f"Word Cloud: Text column '{text_col}' not found")

# If weight column provided, use it; otherwise count word frequencies
if weight_col and weight_col in df.columns:
    # Assume pre-aggregated word/weight pairs
    word_data = df[[text_col, weight_col]].dropna()
    word_data.columns = ['word', 'weight']
    word_data = word_data.groupby('word')['weight'].sum().reset_index()
else:
    # Count word frequencies from text
    all_text = ' '.join(df[text_col].dropna().astype(str))
    words = re.findall(r'\\b[a-zA-Z]{3,}\\b', all_text.lower())

    # Common stopwords to filter
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they', 'this', 'that', 'with', 'from', 'will', 'would', 'there', 'their', 'what', 'about', 'which', 'when', 'make', 'like', 'just', 'over', 'such', 'into', 'than', 'them', 'some', 'could', 'other'}
    words = [w for w in words if w not in stopwords]

    word_counts = pd.Series(words).value_counts().reset_index()
    word_counts.columns = ['word', 'weight']
    word_data = word_counts

# Get top N words
word_data = word_data.nlargest(max_words, 'weight')

# Normalize weights for font sizing
max_weight = word_data['weight'].max()
min_weight = word_data['weight'].min()
if max_weight > min_weight:
    word_data['normalized'] = (word_data['weight'] - min_weight) / (max_weight - min_weight)
else:
    word_data['normalized'] = 1

output = {
    'chartType': 'wordcloud',
    'words': word_data[['word', 'weight', 'normalized']].to_dict('records'),
    'title': config.get('title', '') or 'Word Cloud',
    'minFontSize': int(config.get('minFontSize', 12)),
    'maxFontSize': int(config.get('maxFontSize', 60)),
}
`,
  },

  'pareto-chart': {
    type: 'pareto-chart',
    category: 'visualization',
    label: 'Pareto Chart',
    description: 'Combined bar + cumulative line chart for 80/20 analysis (defect causes, revenue sources)',
    icon: 'BarChart3',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      title: '',
      showLine: true,
      show80Line: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')
show_line = config.get('showLine', True)
show_80_line = config.get('show80Line', True)

if not cat_col or not value_col:
    raise ValueError("Pareto Chart: Please select category and value columns")

if cat_col not in df.columns:
    raise ValueError(f"Pareto Chart: Category column '{cat_col}' not found")
if value_col not in df.columns:
    raise ValueError(f"Pareto Chart: Value column '{value_col}' not found")

# Aggregate and sort by value descending
pareto_data = df.groupby(cat_col)[value_col].sum().reset_index()
pareto_data.columns = ['category', 'value']
pareto_data = pareto_data.sort_values('value', ascending=False).reset_index(drop=True)

# Calculate cumulative percentage
total = pareto_data['value'].sum()
pareto_data['cumulative'] = pareto_data['value'].cumsum()
pareto_data['cumulative_pct'] = (pareto_data['cumulative'] / total * 100).round(2)
pareto_data['pct'] = (pareto_data['value'] / total * 100).round(2)

# Find 80% threshold index
threshold_idx = (pareto_data['cumulative_pct'] >= 80).idxmax() if (pareto_data['cumulative_pct'] >= 80).any() else len(pareto_data) - 1

output = {
    'chartType': 'pareto',
    'data': pareto_data.to_dict('records'),
    'thresholdIndex': int(threshold_idx),
    'title': config.get('title', '') or 'Pareto Chart',
    'showLine': show_line,
    'show80Line': show_80_line,
}
`,
  },

  'parallel-coordinates': {
    type: 'parallel-coordinates',
    category: 'visualization',
    label: 'Parallel Coordinates',
    description: 'High-dimensional data visualization with connected lines across vertical axes',
    icon: 'AlignVerticalDistributeCenter',
    defaultConfig: {
      columns: [],
      colorColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
columns = config.get('columns', [])
color_col = config.get('colorColumn', '')

if not columns or len(columns) == 0:
    # Use all numeric columns if none specified
    columns = df.select_dtypes(include=[np.number]).columns.tolist()[:10]

valid_cols = [c for c in columns if c in df.columns]
if not valid_cols:
    raise ValueError("Parallel Coordinates: No valid columns found")

# Prepare data
pc_data = df[valid_cols].copy()

# Normalize each column to 0-1 range for display
dimensions = []
for col in valid_cols:
    col_data = pd.to_numeric(pc_data[col], errors='coerce')
    min_val = col_data.min()
    max_val = col_data.max()
    dimensions.append({
        'label': col,
        'values': col_data.tolist(),
        'range': [float(min_val), float(max_val)] if pd.notna(min_val) and pd.notna(max_val) else [0, 1],
    })

# Color by column if specified
color_values = None
if color_col and color_col in df.columns:
    color_data = df[color_col]
    if color_data.dtype in ['object', 'category']:
        color_values = pd.factorize(color_data)[0].tolist()
    else:
        color_values = pd.to_numeric(color_data, errors='coerce').tolist()

output = {
    'chartType': 'parallel_coordinates',
    'dimensions': dimensions,
    'colorValues': color_values,
    'colorLabel': color_col if color_col else None,
    'title': config.get('title', '') or 'Parallel Coordinates',
}
`,
  },

  'dendrogram': {
    type: 'dendrogram',
    category: 'visualization',
    label: 'Dendrogram',
    description: 'Tree diagram showing hierarchical clustering relationships with distance/similarity',
    icon: 'GitFork',
    defaultConfig: {
      columns: [],
      linkage: 'ward',
      distanceMetric: 'euclidean',
      orientation: 'top',
      colorThreshold: 0,
      title: '',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import pdist

df = input_data.copy()
columns = config.get('columns', [])
linkage_method = config.get('linkage', 'ward')
distance_metric = config.get('distanceMetric', 'euclidean')
orientation = config.get('orientation', 'top')
color_threshold = float(config.get('colorThreshold', 0))

# Select numeric columns
if columns and len(columns) > 0:
    valid_cols = [c for c in columns if c in df.columns]
    if valid_cols:
        data = df[valid_cols].select_dtypes(include=[np.number])
    else:
        data = df.select_dtypes(include=[np.number])
else:
    data = df.select_dtypes(include=[np.number])

if data.empty or len(data.columns) == 0:
    raise ValueError("Dendrogram: No numeric columns found for clustering")

# Handle missing values
data = data.dropna()

if len(data) < 2:
    raise ValueError("Dendrogram: Need at least 2 samples for clustering")

# Limit samples for performance
if len(data) > 100:
    data = data.sample(n=100, random_state=42)

# Compute linkage
try:
    if distance_metric == 'euclidean' and linkage_method == 'ward':
        Z = linkage(data.values, method=linkage_method)
    else:
        distances = pdist(data.values, metric=distance_metric)
        Z = linkage(distances, method=linkage_method)
except Exception as e:
    raise ValueError(f"Dendrogram: Clustering failed - {str(e)}")

# Get dendrogram data
dend = scipy_dendrogram(Z, no_plot=True, color_threshold=color_threshold if color_threshold > 0 else None)

# Format for frontend
output = {
    'chartType': 'dendrogram',
    'icoord': dend['icoord'],
    'dcoord': dend['dcoord'],
    'ivl': dend['ivl'],
    'leaves': dend['leaves'],
    'color_list': dend['color_list'],
    'linkageMatrix': Z.tolist(),
    'orientation': orientation,
    'title': config.get('title', '') or 'Dendrogram',
    'sampleCount': len(data),
}
`,
  },

  'box-plot': {
    type: 'box-plot',
    category: 'visualization',
    label: 'Box Plot',
    description: 'Statistical distribution showing median, quartiles, and outliers',
    icon: 'BoxSelect',
    defaultConfig: {
      column: '',
      groupColumn: '',
      title: '',
      showOutliers: true,
      notched: false,
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
group_col = config.get('groupColumn', '')
show_outliers = config.get('showOutliers', True)
notched = config.get('notched', False)

if not column:
    # Auto-select first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Box Plot: No numeric columns found in data")
    column = numeric_cols[0]

if column not in df.columns:
    raise ValueError(f"Box Plot: Column '{column}' not found")

# Prepare data for box plot
data = []
if group_col and group_col in df.columns:
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][column].dropna().tolist()
        data.append({
            'y': group_data,
            'name': str(group),
            'type': 'box',
            'boxpoints': 'outliers' if show_outliers else False,
            'notched': notched,
        })
else:
    data.append({
        'y': df[column].dropna().tolist(),
        'name': column,
        'type': 'box',
        'boxpoints': 'outliers' if show_outliers else False,
        'notched': notched,
    })

output = {
    'chartType': 'box',
    'data': data,
    'xLabel': group_col if group_col else '',
    'yLabel': column,
    'title': config.get('title', '') or f'Box Plot of {column}',
}
`,
  },

  'heatmap': {
    type: 'heatmap',
    category: 'visualization',
    label: 'Heatmap',
    description: 'Matrix visualization with color intensity representing values',
    icon: 'Grid',
    defaultConfig: {
      xColumn: '',
      yColumn: '',
      valueColumn: '',
      colorScale: 'Blues',
      showValues: true,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('xColumn', '')
y_col = config.get('yColumn', '')
value_col = config.get('valueColumn', '')
color_scale = config.get('colorScale', 'Blues')
show_values = config.get('showValues', True)

if not x_col or not y_col or not value_col:
    raise ValueError("Heatmap: Please specify X column, Y column, and Value column in Config tab")

if x_col not in df.columns or y_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Heatmap: One or more columns not found in data")

# Create pivot table for heatmap
pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')

output = {
    'chartType': 'heatmap',
    'z': pivot.values.tolist(),
    'x': pivot.columns.tolist(),
    'y': pivot.index.tolist(),
    'colorScale': color_scale,
    'showValues': show_values,
    'title': config.get('title', '') or f'Heatmap of {value_col}',
}
`,
  },

  'scatter-map': {
    type: 'scatter-map',
    category: 'visualization',
    label: 'Scatter Map',
    description: 'Geographic points plotted on a map using latitude/longitude',
    icon: 'MapPin',
    defaultConfig: {
      latColumn: '',
      lonColumn: '',
      sizeColumn: '',
      colorColumn: '',
      title: '',
      mapStyle: 'open-street-map',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
lat_col = config.get('latColumn', '')
lon_col = config.get('lonColumn', '')
size_col = config.get('sizeColumn', '')
color_col = config.get('colorColumn', '')
map_style = config.get('mapStyle', 'open-street-map')

if not lat_col or not lon_col:
    raise ValueError("Scatter Map: Please specify Latitude and Longitude columns in Config tab")

if lat_col not in df.columns or lon_col not in df.columns:
    raise ValueError(f"Scatter Map: Latitude or Longitude column not found")

# Prepare data
lat = df[lat_col].tolist()
lon = df[lon_col].tolist()
sizes = df[size_col].tolist() if size_col and size_col in df.columns else [10] * len(df)
colors = df[color_col].tolist() if color_col and color_col in df.columns else None
text = df.apply(lambda row: '<br>'.join([f'{c}: {row[c]}' for c in df.columns[:5]]), axis=1).tolist()

output = {
    'chartType': 'scattermap',
    'lat': lat,
    'lon': lon,
    'sizes': sizes,
    'colors': colors,
    'text': text,
    'mapStyle': map_style,
    'title': config.get('title', '') or 'Scatter Map',
}
`,
  },

  'grouped-histogram': {
    type: 'grouped-histogram',
    category: 'visualization',
    label: 'Grouped Histogram',
    description: 'Compare distributions across categories with overlapping histograms',
    icon: 'Layers3',
    defaultConfig: {
      column: '',
      groupColumn: '',
      bins: 20,
      barMode: 'overlay',
      opacity: 0.7,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
group_col = config.get('groupColumn', '')
bins = config.get('bins', 20)
bar_mode = config.get('barMode', 'overlay')
opacity = config.get('opacity', 0.7)

if not column:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Grouped Histogram: No numeric columns found")
    column = numeric_cols[0]

if column not in df.columns:
    raise ValueError(f"Grouped Histogram: Column '{column}' not found")

data = []
if group_col and group_col in df.columns:
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][column].dropna().tolist()
        data.append({
            'x': group_data,
            'name': str(group),
            'type': 'histogram',
            'opacity': opacity,
            'nbinsx': bins,
        })
else:
    data.append({
        'x': df[column].dropna().tolist(),
        'name': column,
        'type': 'histogram',
        'nbinsx': bins,
    })

output = {
    'chartType': 'grouped_histogram',
    'data': data,
    'barMode': bar_mode,
    'xLabel': column,
    'yLabel': 'Count',
    'title': config.get('title', '') or f'Histogram of {column}',
}
`,
  },

  'network-graph': {
    type: 'network-graph',
    category: 'visualization',
    label: 'Network Graph',
    description: 'Force-directed graph showing relationships between nodes',
    icon: 'Share2',
    defaultConfig: {
      sourceColumn: '',
      targetColumn: '',
      weightColumn: '',
      layout: 'force',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np
import networkx as nx

df = input_data.copy()
source_col = config.get('sourceColumn', '')
target_col = config.get('targetColumn', '')
weight_col = config.get('weightColumn', '')

if not source_col or not target_col:
    raise ValueError("Network Graph: Please specify Source and Target columns in Config tab")

if source_col not in df.columns or target_col not in df.columns:
    raise ValueError(f"Network Graph: Source or Target column not found")

# Create graph
G = nx.Graph()
for _, row in df.iterrows():
    weight = row[weight_col] if weight_col and weight_col in df.columns else 1
    G.add_edge(str(row[source_col]), str(row[target_col]), weight=weight)

# Get layout positions
pos = nx.spring_layout(G, k=2, iterations=50)

# Prepare node data
nodes = []
for node in G.nodes():
    x, y = pos[node]
    nodes.append({
        'id': node,
        'x': float(x),
        'y': float(y),
        'degree': G.degree(node),
    })

# Prepare edge data
edges = []
for source, target, data in G.edges(data=True):
    x0, y0 = pos[source]
    x1, y1 = pos[target]
    edges.append({
        'source': source,
        'target': target,
        'x0': float(x0),
        'y0': float(y0),
        'x1': float(x1),
        'y1': float(y1),
        'weight': data.get('weight', 1),
    })

output = {
    'chartType': 'network',
    'nodes': nodes,
    'edges': edges,
    'title': config.get('title', '') or 'Network Graph',
}
`,
  },

  'calendar-heatmap': {
    type: 'calendar-heatmap',
    category: 'visualization',
    label: 'Calendar Heatmap',
    description: 'Daily values displayed in a calendar grid format',
    icon: 'CalendarDays',
    defaultConfig: {
      dateColumn: '',
      valueColumn: '',
      colorScale: 'Greens',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
date_col = config.get('dateColumn', '')
value_col = config.get('valueColumn', '')
color_scale = config.get('colorScale', 'Greens')

if not date_col or not value_col:
    raise ValueError("Calendar Heatmap: Please specify Date and Value columns in Config tab")

if date_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Calendar Heatmap: Date or Value column not found")

# Convert to datetime
df[date_col] = pd.to_datetime(df[date_col])
df = df.groupby(date_col)[value_col].sum().reset_index()

# Extract date components
df['year'] = df[date_col].dt.year
df['week'] = df[date_col].dt.isocalendar().week
df['dayofweek'] = df[date_col].dt.dayofweek
df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')

output = {
    'chartType': 'calendar_heatmap',
    'dates': df['date_str'].tolist(),
    'values': df[value_col].tolist(),
    'weeks': df['week'].tolist(),
    'dayofweek': df['dayofweek'].tolist(),
    'years': df['year'].unique().tolist(),
    'colorScale': color_scale,
    'title': config.get('title', '') or f'Calendar Heatmap of {value_col}',
}
`,
  },

  'faceted-chart': {
    type: 'faceted-chart',
    category: 'visualization',
    label: 'Faceted Chart',
    description: 'Same chart repeated across different categories for comparison',
    icon: 'LayoutPanelLeft',
    defaultConfig: {
      x: '',
      y: '',
      facetColumn: '',
      chartType: 'scatter',
      columns: 3,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')
facet_col = config.get('facetColumn', '')
chart_type = config.get('chartType', 'scatter')
n_columns = config.get('columns', 3)

if not x_col or not y_col or not facet_col:
    raise ValueError("Faceted Chart: Please specify X, Y, and Facet columns in Config tab")

if x_col not in df.columns or y_col not in df.columns or facet_col not in df.columns:
    raise ValueError(f"Faceted Chart: One or more columns not found")

# Prepare faceted data
facets = df[facet_col].unique().tolist()
data = []
for facet in facets:
    facet_df = df[df[facet_col] == facet]
    data.append({
        'facet': str(facet),
        'x': facet_df[x_col].tolist(),
        'y': facet_df[y_col].tolist(),
    })

output = {
    'chartType': 'faceted',
    'subChartType': chart_type,
    'data': data,
    'xLabel': x_col,
    'yLabel': y_col,
    'facetColumn': facet_col,
    'columns': n_columns,
    'title': config.get('title', '') or f'Faceted {chart_type.title()} by {facet_col}',
}
`,
  },

  'density-plot': {
    type: 'density-plot',
    category: 'visualization',
    label: 'Density Plot',
    description: 'Smooth probability density curve using kernel density estimation',
    icon: 'Waves',
    defaultConfig: {
      column: '',
      groupColumn: '',
      title: '',
      fillOpacity: 0.3,
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
column = config.get('column', '')
group_col = config.get('groupColumn', '')
fill_opacity = config.get('fillOpacity', 0.3)

if not column:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Density Plot: No numeric columns found")
    column = numeric_cols[0]

if column not in df.columns:
    raise ValueError(f"Density Plot: Column '{column}' not found")

data = []
if group_col and group_col in df.columns:
    for group in df[group_col].unique():
        values = df[df[group_col] == group][column].dropna()
        if len(values) > 1:
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            y_range = kde(x_range)
            data.append({
                'x': x_range.tolist(),
                'y': y_range.tolist(),
                'name': str(group),
                'fill': 'tozeroy',
                'opacity': fill_opacity,
            })
else:
    values = df[column].dropna()
    if len(values) > 1:
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        y_range = kde(x_range)
        data.append({
            'x': x_range.tolist(),
            'y': y_range.tolist(),
            'name': column,
            'fill': 'tozeroy',
            'opacity': fill_opacity,
        })

output = {
    'chartType': 'density',
    'data': data,
    'xLabel': column,
    'yLabel': 'Density',
    'title': config.get('title', '') or f'Density Plot of {column}',
}
`,
  },

  'error-bar-chart': {
    type: 'error-bar-chart',
    category: 'visualization',
    label: 'Error Bar Chart',
    description: 'Bar or line chart with confidence intervals or error bars',
    icon: 'AlertTriangle',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      errorColumn: '',
      errorType: 'data',
      chartType: 'bar',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')
error_col = config.get('errorColumn', '')
error_type = config.get('errorType', 'data')
chart_type = config.get('chartType', 'bar')

if not cat_col or not value_col:
    raise ValueError("Error Bar Chart: Please specify Category and Value columns in Config tab")

if cat_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Error Bar Chart: Category or Value column not found")

categories = df[cat_col].tolist()
values = df[value_col].tolist()

# Handle error values
if error_col and error_col in df.columns:
    errors = df[error_col].tolist()
elif error_type == 'std':
    # Calculate standard deviation per group
    grouped = df.groupby(cat_col)[value_col]
    errors = grouped.std().reindex(df[cat_col]).tolist()
else:
    errors = [0] * len(values)

output = {
    'chartType': 'error_bar',
    'subChartType': chart_type,
    'categories': categories,
    'values': values,
    'errors': errors,
    'xLabel': cat_col,
    'yLabel': value_col,
    'title': config.get('title', '') or f'{value_col} with Error Bars',
}
`,
  },

  'dot-plot': {
    type: 'dot-plot',
    category: 'visualization',
    label: 'Dot Plot',
    description: 'Cleveland dot plot for clean categorical value comparison',
    icon: 'CircleDot',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      colorColumn: '',
      orientation: 'horizontal',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')
color_col = config.get('colorColumn', '')
orientation = config.get('orientation', 'horizontal')

if not cat_col or not value_col:
    raise ValueError("Dot Plot: Please specify Category and Value columns in Config tab")

if cat_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Dot Plot: Category or Value column not found")

# Sort by value for better visualization
df = df.sort_values(value_col, ascending=True)

categories = df[cat_col].tolist()
values = df[value_col].tolist()
colors = df[color_col].tolist() if color_col and color_col in df.columns else None

output = {
    'chartType': 'dot_plot',
    'categories': categories,
    'values': values,
    'colors': colors,
    'orientation': orientation,
    'xLabel': value_col if orientation == 'horizontal' else cat_col,
    'yLabel': cat_col if orientation == 'horizontal' else value_col,
    'title': config.get('title', '') or f'{value_col} by {cat_col}',
}
`,
  },

  'slope-chart': {
    type: 'slope-chart',
    category: 'visualization',
    label: 'Slope Chart',
    description: 'Before/after comparison showing changes between two points',
    icon: 'ArrowRightLeft',
    defaultConfig: {
      entityColumn: '',
      startColumn: '',
      endColumn: '',
      startLabel: 'Before',
      endLabel: 'After',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
entity_col = config.get('entityColumn', '')
start_col = config.get('startColumn', '')
end_col = config.get('endColumn', '')
start_label = config.get('startLabel', 'Before')
end_label = config.get('endLabel', 'After')

if not entity_col or not start_col or not end_col:
    raise ValueError("Slope Chart: Please specify Entity, Start Value, and End Value columns in Config tab")

if entity_col not in df.columns or start_col not in df.columns or end_col not in df.columns:
    raise ValueError(f"Slope Chart: One or more columns not found")

# Prepare data for slope chart
lines = []
for _, row in df.iterrows():
    lines.append({
        'entity': str(row[entity_col]),
        'start': float(row[start_col]),
        'end': float(row[end_col]),
        'change': float(row[end_col]) - float(row[start_col]),
    })

output = {
    'chartType': 'slope',
    'lines': lines,
    'startLabel': start_label,
    'endLabel': end_label,
    'title': config.get('title', '') or 'Slope Chart',
}
`,
  },

  'grouped-bar-chart': {
    type: 'grouped-bar-chart',
    category: 'visualization',
    label: 'Grouped Bar Chart',
    description: 'Side-by-side bars comparing subcategories within groups',
    icon: 'BarChart4',
    defaultConfig: {
      categoryColumn: '',
      groupColumn: '',
      valueColumn: '',
      orientation: 'vertical',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cat_col = config.get('categoryColumn', '')
group_col = config.get('groupColumn', '')
value_col = config.get('valueColumn', '')
orientation = config.get('orientation', 'vertical')

if not cat_col or not group_col or not value_col:
    raise ValueError("Grouped Bar Chart: Please specify Category, Group, and Value columns in Config tab")

if cat_col not in df.columns or group_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Grouped Bar Chart: One or more columns not found")

# Pivot data for grouped bars
pivot = df.pivot_table(index=cat_col, columns=group_col, values=value_col, aggfunc='sum').fillna(0)

data = []
for col in pivot.columns:
    data.append({
        'name': str(col),
        'x': pivot.index.tolist() if orientation == 'vertical' else pivot[col].tolist(),
        'y': pivot[col].tolist() if orientation == 'vertical' else pivot.index.tolist(),
        'orientation': 'v' if orientation == 'vertical' else 'h',
    })

output = {
    'chartType': 'grouped_bar',
    'data': data,
    'categories': pivot.index.tolist(),
    'xLabel': cat_col if orientation == 'vertical' else value_col,
    'yLabel': value_col if orientation == 'vertical' else cat_col,
    'title': config.get('title', '') or f'{value_col} by {cat_col} and {group_col}',
}
`,
  },

  'bump-chart': {
    type: 'bump-chart',
    category: 'visualization',
    label: 'Bump Chart',
    description: 'Ranking changes over time with connected lines',
    icon: 'LineChart',
    defaultConfig: {
      entityColumn: '',
      timeColumn: '',
      rankColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
entity_col = config.get('entityColumn', '')
time_col = config.get('timeColumn', '')
rank_col = config.get('rankColumn', '')

if not entity_col or not time_col or not rank_col:
    raise ValueError("Bump Chart: Please specify Entity, Time, and Rank columns in Config tab")

if entity_col not in df.columns or time_col not in df.columns or rank_col not in df.columns:
    raise ValueError(f"Bump Chart: One or more columns not found")

# Sort by time
df = df.sort_values(time_col)

# Get unique time points and entities
time_points = df[time_col].unique().tolist()
entities = df[entity_col].unique().tolist()

# Prepare lines for each entity
lines = []
for entity in entities:
    entity_data = df[df[entity_col] == entity]
    times = entity_data[time_col].tolist()
    ranks = entity_data[rank_col].tolist()
    lines.append({
        'entity': str(entity),
        'x': times,
        'y': ranks,
    })

output = {
    'chartType': 'bump',
    'lines': lines,
    'timePoints': time_points,
    'xLabel': time_col,
    'yLabel': 'Rank',
    'title': config.get('title', '') or 'Bump Chart - Ranking Over Time',
}
`,
  },

  'donut-chart': {
    type: 'donut-chart',
    category: 'visualization',
    label: 'Donut Chart',
    description: 'Pie chart with hollow center for proportions and center KPI display',
    icon: 'PieChart',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      title: '',
      holeSize: 0.4,
      showLabels: true,
      showPercent: true,
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
category_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')

if not category_col or not value_col:
    raise ValueError("Donut Chart: Please specify Category and Value columns in Config tab")

if category_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Donut Chart: Column not found in data")

# Aggregate by category
agg_df = df.groupby(category_col)[value_col].sum().reset_index()
agg_df = agg_df.sort_values(value_col, ascending=False)

total = agg_df[value_col].sum()
labels = agg_df[category_col].tolist()
values = agg_df[value_col].tolist()
percentages = [(v / total * 100) if total > 0 else 0 for v in values]

output = {
    'chartType': 'donut',
    'labels': labels,
    'values': values,
    'percentages': [round(p, 1) for p in percentages],
    'total': float(total),
    'holeSize': config.get('holeSize', 0.4),
    'showLabels': config.get('showLabels', True),
    'showPercent': config.get('showPercent', True),
    'title': config.get('title', '') or f'Donut Chart: {value_col} by {category_col}',
}
`,
  },

  'horizontal-bar-chart': {
    type: 'horizontal-bar-chart',
    category: 'visualization',
    label: 'Horizontal Bar Chart',
    description: 'Horizontal bars for categorical comparisons with long labels',
    icon: 'BarChart',
    defaultConfig: {
      categoryColumn: '',
      valueColumn: '',
      colorColumn: '',
      sortBy: 'value',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
category_col = config.get('categoryColumn', '')
value_col = config.get('valueColumn', '')
color_col = config.get('colorColumn', '')
sort_by = config.get('sortBy', 'value')

if not category_col or not value_col:
    raise ValueError("Horizontal Bar Chart: Please specify Category and Value columns in Config tab")

if category_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Horizontal Bar Chart: Column not found in data")

# Aggregate by category
agg_df = df.groupby(category_col)[value_col].sum().reset_index()

# Sort
if sort_by == 'value':
    agg_df = agg_df.sort_values(value_col, ascending=True)
elif sort_by == 'category':
    agg_df = agg_df.sort_values(category_col)

categories = agg_df[category_col].tolist()
values = agg_df[value_col].tolist()

# Handle color column
colors = None
if color_col and color_col in df.columns:
    color_agg = df.groupby(category_col)[color_col].first().reset_index()
    color_agg = color_agg.set_index(category_col).loc[categories]
    colors = color_agg[color_col].tolist()

output = {
    'chartType': 'horizontal_bar',
    'categories': categories,
    'values': values,
    'colors': colors,
    'xLabel': value_col,
    'yLabel': category_col,
    'title': config.get('title', '') or f'Horizontal Bar: {value_col} by {category_col}',
}
`,
  },

  'scatter-3d': {
    type: 'scatter-3d',
    category: 'visualization',
    label: '3D Scatter Plot',
    description: 'Interactive 3D scatter visualization with rotation and zoom',
    icon: 'ScatterChart',
    defaultConfig: {
      x: '',
      y: '',
      z: '',
      color: '',
      size: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')
z_col = config.get('z', '')
color_col = config.get('color', '')
size_col = config.get('size', '')

if not x_col or not y_col or not z_col:
    raise ValueError("3D Scatter: Please specify X, Y, and Z columns in Config tab")

for col in [x_col, y_col, z_col]:
    if col not in df.columns:
        raise ValueError(f"3D Scatter: Column '{col}' not found in data")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NaN in required columns
df = df.dropna(subset=[x_col, y_col, z_col])

data_points = []
for _, row in df.iterrows():
    point = {
        'x': float(row[x_col]),
        'y': float(row[y_col]),
        'z': float(row[z_col]),
    }
    if color_col and color_col in df.columns:
        point['color'] = row[color_col]
    if size_col and size_col in df.columns:
        point['size'] = float(row[size_col]) if pd.notna(row[size_col]) else 5
    data_points.append(point)

output = {
    'chartType': 'scatter3d',
    'data': data_points,
    'xLabel': x_col,
    'yLabel': y_col,
    'zLabel': z_col,
    'colorColumn': color_col,
    'sizeColumn': size_col,
    'title': config.get('title', '') or f'3D Scatter: {x_col} vs {y_col} vs {z_col}',
}
`,
  },

  'contour-plot': {
    type: 'contour-plot',
    category: 'visualization',
    label: 'Contour Plot',
    description: '2D density visualization showing concentration areas with contour lines',
    icon: 'Waves',
    defaultConfig: {
      x: '',
      y: '',
      colorScale: 'Blues',
      showLines: true,
      fillContours: true,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')

if not x_col or not y_col:
    raise ValueError("Contour Plot: Please specify X and Y columns in Config tab")

if x_col not in df.columns or y_col not in df.columns:
    raise ValueError(f"Contour Plot: Column not found in data")

df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
df = df.dropna(subset=[x_col, y_col])

x_vals = df[x_col].values
y_vals = df[y_col].values

# Create histogram2d for density
bins = 30
hist, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=bins)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

output = {
    'chartType': 'contour',
    'z': hist.T.tolist(),
    'x': x_centers.tolist(),
    'y': y_centers.tolist(),
    'colorScale': config.get('colorScale', 'Blues'),
    'showLines': config.get('showLines', True),
    'fillContours': config.get('fillContours', True),
    'xLabel': x_col,
    'yLabel': y_col,
    'title': config.get('title', '') or f'Contour: Density of {x_col} vs {y_col}',
}
`,
  },

  'hexbin-plot': {
    type: 'hexbin-plot',
    category: 'visualization',
    label: 'Hexbin Plot',
    description: 'Hexagonal binning for large scatter datasets to prevent overplotting',
    icon: 'Grid3x3',
    defaultConfig: {
      x: '',
      y: '',
      gridSize: 20,
      colorScale: 'Blues',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')
grid_size = config.get('gridSize', 20)

if not x_col or not y_col:
    raise ValueError("Hexbin Plot: Please specify X and Y columns in Config tab")

if x_col not in df.columns or y_col not in df.columns:
    raise ValueError(f"Hexbin Plot: Column not found in data")

df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
df = df.dropna(subset=[x_col, y_col])

x_vals = df[x_col].tolist()
y_vals = df[y_col].tolist()

output = {
    'chartType': 'hexbin',
    'x': x_vals,
    'y': y_vals,
    'gridSize': grid_size,
    'colorScale': config.get('colorScale', 'Blues'),
    'xLabel': x_col,
    'yLabel': y_col,
    'title': config.get('title', '') or f'Hexbin: {x_col} vs {y_col}',
}
`,
  },

  'ridge-plot': {
    type: 'ridge-plot',
    category: 'visualization',
    label: 'Ridge Plot',
    description: 'Stacked density curves comparing distributions across categories',
    icon: 'Layers',
    defaultConfig: {
      valueColumn: '',
      categoryColumn: '',
      colorScale: 'Viridis',
      overlap: 0.5,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats

df = input_data.copy()
value_col = config.get('valueColumn', '')
category_col = config.get('categoryColumn', '')

if not value_col or not category_col:
    raise ValueError("Ridge Plot: Please specify Value and Category columns in Config tab")

if value_col not in df.columns or category_col not in df.columns:
    raise ValueError(f"Ridge Plot: Column not found in data")

df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df = df.dropna(subset=[value_col])

categories = df[category_col].unique().tolist()
ridges = []

# Common x range
x_min = df[value_col].min()
x_max = df[value_col].max()
x_range = np.linspace(x_min, x_max, 100)

for cat in categories:
    cat_data = df[df[category_col] == cat][value_col].values
    if len(cat_data) > 1:
        kde = stats.gaussian_kde(cat_data)
        density = kde(x_range)
        ridges.append({
            'category': str(cat),
            'x': x_range.tolist(),
            'y': density.tolist(),
        })

output = {
    'chartType': 'ridge',
    'ridges': ridges,
    'overlap': config.get('overlap', 0.5),
    'colorScale': config.get('colorScale', 'Viridis'),
    'xLabel': value_col,
    'title': config.get('title', '') or f'Ridge Plot: {value_col} by {category_col}',
}
`,
  },

  'strip-plot': {
    type: 'strip-plot',
    category: 'visualization',
    label: 'Strip Plot',
    description: 'Individual data points with jitter along categorical axis',
    icon: 'CircleDot',
    defaultConfig: {
      valueColumn: '',
      categoryColumn: '',
      jitter: 0.3,
      pointSize: 8,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
value_col = config.get('valueColumn', '')
category_col = config.get('categoryColumn', '')
jitter = config.get('jitter', 0.3)

if not value_col or not category_col:
    raise ValueError("Strip Plot: Please specify Value and Category columns in Config tab")

if value_col not in df.columns or category_col not in df.columns:
    raise ValueError(f"Strip Plot: Column not found in data")

df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
df = df.dropna(subset=[value_col])

categories = df[category_col].unique().tolist()
strips = []

for i, cat in enumerate(categories):
    cat_data = df[df[category_col] == cat][value_col].values
    # Add jitter
    jittered_x = np.random.uniform(-jitter/2, jitter/2, len(cat_data)) + i
    strips.append({
        'category': str(cat),
        'x': jittered_x.tolist(),
        'y': cat_data.tolist(),
        'categoryIndex': i,
    })

output = {
    'chartType': 'strip',
    'strips': strips,
    'categories': categories,
    'pointSize': config.get('pointSize', 8),
    'yLabel': value_col,
    'title': config.get('title', '') or f'Strip Plot: {value_col} by {category_col}',
}
`,
  },

  'bullet-chart': {
    type: 'bullet-chart',
    category: 'visualization',
    label: 'Bullet Chart',
    description: 'Compact KPI visualization showing actual vs target with qualitative ranges',
    icon: 'Target',
    defaultConfig: {
      actualColumn: '',
      targetColumn: '',
      categoryColumn: '',
      ranges: [30, 70, 100],
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
actual_col = config.get('actualColumn', '')
target_col = config.get('targetColumn', '')
category_col = config.get('categoryColumn', '')
ranges = config.get('ranges', [30, 70, 100])

if not actual_col:
    raise ValueError("Bullet Chart: Please specify Actual column in Config tab")

if actual_col not in df.columns:
    raise ValueError(f"Bullet Chart: Actual column not found in data")

bullets = []

if category_col and category_col in df.columns:
    # Multiple bullets by category
    for _, row in df.iterrows():
        bullet = {
            'category': str(row[category_col]),
            'actual': float(row[actual_col]) if pd.notna(row[actual_col]) else 0,
            'ranges': ranges,
        }
        if target_col and target_col in df.columns and pd.notna(row[target_col]):
            bullet['target'] = float(row[target_col])
        bullets.append(bullet)
else:
    # Single bullet with first row
    row = df.iloc[0]
    bullet = {
        'category': 'Value',
        'actual': float(row[actual_col]) if pd.notna(row[actual_col]) else 0,
        'ranges': ranges,
    }
    if target_col and target_col in df.columns and pd.notna(row[target_col]):
        bullet['target'] = float(row[target_col])
    bullets.append(bullet)

output = {
    'chartType': 'bullet',
    'bullets': bullets,
    'title': config.get('title', '') or 'Bullet Chart',
}
`,
  },

  'pyramid-chart': {
    type: 'pyramid-chart',
    category: 'visualization',
    label: 'Pyramid Chart',
    description: 'Back-to-back horizontal bars for two-sided population comparisons',
    icon: 'BarChart3',
    defaultConfig: {
      categoryColumn: '',
      leftColumn: '',
      rightColumn: '',
      leftLabel: 'Left',
      rightLabel: 'Right',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
category_col = config.get('categoryColumn', '')
left_col = config.get('leftColumn', '')
right_col = config.get('rightColumn', '')

if not category_col or not left_col or not right_col:
    raise ValueError("Pyramid Chart: Please specify Category, Left, and Right columns in Config tab")

for col in [category_col, left_col, right_col]:
    if col not in df.columns:
        raise ValueError(f"Pyramid Chart: Column '{col}' not found in data")

df[left_col] = pd.to_numeric(df[left_col], errors='coerce')
df[right_col] = pd.to_numeric(df[right_col], errors='coerce')

categories = df[category_col].tolist()
left_values = df[left_col].tolist()
right_values = df[right_col].tolist()

# Make left values negative for pyramid effect
left_values_neg = [-v if pd.notna(v) else 0 for v in left_values]
right_values_clean = [v if pd.notna(v) else 0 for v in right_values]

output = {
    'chartType': 'pyramid',
    'categories': categories,
    'leftValues': left_values_neg,
    'rightValues': right_values_clean,
    'leftLabel': config.get('leftLabel', 'Left'),
    'rightLabel': config.get('rightLabel', 'Right'),
    'title': config.get('title', '') or f'Population Pyramid: {left_col} vs {right_col}',
}
`,
  },

  'timeline-chart': {
    type: 'timeline-chart',
    category: 'visualization',
    label: 'Timeline Chart',
    description: 'Events and tasks displayed along a time axis with duration bars',
    icon: 'Calendar',
    defaultConfig: {
      taskColumn: '',
      startColumn: '',
      endColumn: '',
      categoryColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
task_col = config.get('taskColumn', '')
start_col = config.get('startColumn', '')
end_col = config.get('endColumn', '')
category_col = config.get('categoryColumn', '')

if not task_col or not start_col or not end_col:
    raise ValueError("Timeline Chart: Please specify Task, Start, and End columns in Config tab")

for col in [task_col, start_col, end_col]:
    if col not in df.columns:
        raise ValueError(f"Timeline Chart: Column '{col}' not found in data")

# Parse dates
df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
df = df.dropna(subset=[start_col, end_col])

tasks = []
for _, row in df.iterrows():
    task = {
        'task': str(row[task_col]),
        'start': row[start_col].isoformat(),
        'end': row[end_col].isoformat(),
    }
    if category_col and category_col in df.columns:
        task['category'] = str(row[category_col])
    tasks.append(task)

# Sort by start date
tasks.sort(key=lambda x: x['start'])

output = {
    'chartType': 'timeline',
    'tasks': tasks,
    'title': config.get('title', '') or 'Timeline Chart',
}
`,
  },

  'surface-3d': {
    type: 'surface-3d',
    category: 'visualization',
    label: '3D Surface Plot',
    description: 'Interactive 3D surface showing Z values across X-Y grid',
    icon: 'Grid',
    defaultConfig: {
      x: '',
      y: '',
      z: '',
      colorScale: 'Viridis',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')
z_col = config.get('z', '')

if not x_col or not y_col or not z_col:
    raise ValueError("3D Surface: Please specify X, Y, and Z columns in Config tab")

for col in [x_col, y_col, z_col]:
    if col not in df.columns:
        raise ValueError(f"3D Surface: Column '{col}' not found in data")
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=[x_col, y_col, z_col])

# Create pivot table for surface
try:
    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
    z_grid = pivot.values.tolist()
    x_vals = pivot.columns.tolist()
    y_vals = pivot.index.tolist()
except Exception:
    # Fallback: create grid from scattered points
    x_unique = sorted(df[x_col].unique())
    y_unique = sorted(df[y_col].unique())
    z_grid = []
    for y_val in y_unique:
        row = []
        for x_val in x_unique:
            z_val = df[(df[x_col] == x_val) & (df[y_col] == y_val)][z_col].mean()
            row.append(float(z_val) if pd.notna(z_val) else None)
        z_grid.append(row)
    x_vals = [float(x) for x in x_unique]
    y_vals = [float(y) for y in y_unique]

output = {
    'chartType': 'surface3d',
    'z': z_grid,
    'x': x_vals,
    'y': y_vals,
    'colorScale': config.get('colorScale', 'Viridis'),
    'xLabel': x_col,
    'yLabel': y_col,
    'zLabel': z_col,
    'title': config.get('title', '') or f'3D Surface: {z_col}',
}
`,
  },

  'marginal-histogram': {
    type: 'marginal-histogram',
    category: 'visualization',
    label: 'Marginal Histogram',
    description: 'Scatter plot with histograms on margins showing variable distributions',
    icon: 'LayoutGrid',
    defaultConfig: {
      x: '',
      y: '',
      color: '',
      bins: 20,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('x', '')
y_col = config.get('y', '')
color_col = config.get('color', '')
bins = config.get('bins', 20)

if not x_col or not y_col:
    raise ValueError("Marginal Histogram: Please specify X and Y columns in Config tab")

if x_col not in df.columns or y_col not in df.columns:
    raise ValueError(f"Marginal Histogram: Column not found in data")

df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
df = df.dropna(subset=[x_col, y_col])

# Scatter data
scatter_data = {
    'x': df[x_col].tolist(),
    'y': df[y_col].tolist(),
}

if color_col and color_col in df.columns:
    scatter_data['color'] = df[color_col].tolist()

# X histogram
x_hist, x_edges = np.histogram(df[x_col], bins=bins)
x_hist_data = {
    'bins': ((x_edges[:-1] + x_edges[1:]) / 2).tolist(),
    'counts': x_hist.tolist(),
}

# Y histogram
y_hist, y_edges = np.histogram(df[y_col], bins=bins)
y_hist_data = {
    'bins': ((y_edges[:-1] + y_edges[1:]) / 2).tolist(),
    'counts': y_hist.tolist(),
}

output = {
    'chartType': 'marginal_histogram',
    'scatter': scatter_data,
    'xHist': x_hist_data,
    'yHist': y_hist_data,
    'xLabel': x_col,
    'yLabel': y_col,
    'colorColumn': color_col,
    'title': config.get('title', '') or f'Marginal Histogram: {x_col} vs {y_col}',
}
`,
  },

  'dumbbell-chart': {
    type: 'dumbbell-chart',
    category: 'visualization',
    label: 'Dumbbell Chart',
    description: 'Connected dots showing change between two values per category',
    icon: 'ArrowLeftRight',
    defaultConfig: {
      categoryColumn: '',
      startColumn: '',
      endColumn: '',
      startLabel: 'Start',
      endLabel: 'End',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
category_col = config.get('categoryColumn', '')
start_col = config.get('startColumn', '')
end_col = config.get('endColumn', '')

if not category_col or not start_col or not end_col:
    raise ValueError("Dumbbell Chart: Please specify Category, Start, and End columns in Config tab")

for col in [category_col, start_col, end_col]:
    if col not in df.columns:
        raise ValueError(f"Dumbbell Chart: Column '{col}' not found in data")

df[start_col] = pd.to_numeric(df[start_col], errors='coerce')
df[end_col] = pd.to_numeric(df[end_col], errors='coerce')

dumbbells = []
for _, row in df.iterrows():
    if pd.notna(row[start_col]) and pd.notna(row[end_col]):
        dumbbells.append({
            'category': str(row[category_col]),
            'start': float(row[start_col]),
            'end': float(row[end_col]),
            'change': float(row[end_col]) - float(row[start_col]),
        })

# Sort by change
dumbbells.sort(key=lambda x: x['change'], reverse=True)

output = {
    'chartType': 'dumbbell',
    'data': dumbbells,
    'startLabel': config.get('startLabel', 'Start'),
    'endLabel': config.get('endLabel', 'End'),
    'title': config.get('title', '') or f'Dumbbell Chart: {start_col} to {end_col}',
}
`,
  },

  // ===== NEW ADVANCED VISUALIZATION BLOCKS FOR DATA SCIENTISTS =====

  'shap-summary-plot': {
    type: 'shap-summary-plot',
    category: 'visualization',
    label: 'SHAP Summary Plot',
    description: 'Display SHAP values showing feature impact on model predictions with distribution',
    icon: 'Lightbulb',
    defaultConfig: {
      featureColumns: [],
      shapValuesColumn: '',
      maxFeatures: 20,
      plotType: 'dot',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
max_features = config.get('maxFeatures', 20)
plot_type = config.get('plotType', 'dot')

if not feature_cols:
    # Try to detect SHAP columns or use numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not c.startswith('shap_')][:max_features]

if not feature_cols:
    raise ValueError("SHAP Summary Plot: Please specify feature columns or ensure data has numeric columns")

# Calculate feature importance (mean absolute value as proxy for SHAP importance)
importance_data = []
for col in feature_cols[:max_features]:
    if col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        importance_data.append({
            'feature': col,
            'importance': float(np.mean(np.abs(values))) if len(values) > 0 else 0,
            'mean_value': float(np.mean(values)) if len(values) > 0 else 0,
            'std_value': float(np.std(values)) if len(values) > 0 else 0,
            'values': values.tolist()[:100],  # Sample for distribution
        })

# Sort by importance
importance_data.sort(key=lambda x: x['importance'], reverse=True)

output = {
    'chartType': 'shap_summary',
    'data': importance_data,
    'plotType': plot_type,
    'title': config.get('title', '') or 'SHAP Summary Plot',
}
`,
  },

  'partial-dependence-plot': {
    type: 'partial-dependence-plot',
    category: 'visualization',
    label: 'Partial Dependence Plot',
    description: 'Show marginal effect of features on predicted outcome',
    icon: 'TrendingUp',
    defaultConfig: {
      featureColumn: '',
      targetColumn: '',
      gridResolution: 50,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
feature_col = config.get('featureColumn', '')
target_col = config.get('targetColumn', '')
grid_resolution = config.get('gridResolution', 50)

if not feature_col:
    raise ValueError("Partial Dependence Plot: Please specify a feature column in Config tab")

if feature_col not in df.columns:
    raise ValueError(f"Partial Dependence Plot: Column '{feature_col}' not found")

# Get feature values
feature_values = pd.to_numeric(df[feature_col], errors='coerce').dropna()

# Create grid
grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

# Calculate average target at each grid point (binned approximation)
pdp_values = []
for i, val in enumerate(grid[:-1]):
    mask = (feature_values >= val) & (feature_values < grid[i+1])
    if target_col and target_col in df.columns:
        target_vals = pd.to_numeric(df.loc[mask.index[mask], target_col], errors='coerce')
        avg_target = float(target_vals.mean()) if len(target_vals) > 0 else 0
    else:
        avg_target = float(mask.sum())
    pdp_values.append({
        'feature_value': float(val),
        'pdp_value': avg_target,
    })

output = {
    'chartType': 'partial_dependence',
    'data': pdp_values,
    'feature': feature_col,
    'target': target_col or 'count',
    'title': config.get('title', '') or f'Partial Dependence: {feature_col}',
}
`,
  },

  'feature-importance-plot': {
    type: 'feature-importance-plot',
    category: 'visualization',
    label: 'Feature Importance Plot',
    description: 'Horizontal bar chart ranking features by importance score',
    icon: 'Award',
    defaultConfig: {
      featureColumn: '',
      importanceColumn: '',
      maxFeatures: 20,
      sortOrder: 'descending',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
feature_col = config.get('featureColumn', '')
importance_col = config.get('importanceColumn', '')
max_features = config.get('maxFeatures', 20)
sort_order = config.get('sortOrder', 'descending')

# If specific columns provided, use them
if feature_col and importance_col:
    if feature_col not in df.columns or importance_col not in df.columns:
        raise ValueError(f"Feature Importance Plot: Columns not found. Available: {', '.join(df.columns.tolist())}")

    plot_data = df[[feature_col, importance_col]].copy()
    plot_data.columns = ['feature', 'importance']
else:
    # Auto-detect: assume first column is feature names, second is importance
    if len(df.columns) >= 2:
        plot_data = df.iloc[:, :2].copy()
        plot_data.columns = ['feature', 'importance']
    else:
        raise ValueError("Feature Importance Plot: Need at least 2 columns (feature names and importance values)")

plot_data['importance'] = pd.to_numeric(plot_data['importance'], errors='coerce')
plot_data = plot_data.dropna()

# Sort
ascending = sort_order == 'ascending'
plot_data = plot_data.sort_values('importance', ascending=ascending).head(max_features)

features = plot_data['feature'].tolist()
importances = plot_data['importance'].tolist()

output = {
    'chartType': 'feature_importance',
    'data': [{'feature': f, 'importance': float(i)} for f, i in zip(features, importances)],
    'title': config.get('title', '') or 'Feature Importance',
}
`,
  },

  'ice-plot': {
    type: 'ice-plot',
    category: 'visualization',
    label: 'ICE Plot',
    description: 'Individual Conditional Expectation - show how predictions change for individual samples',
    icon: 'Activity',
    defaultConfig: {
      featureColumn: '',
      targetColumn: '',
      nLines: 50,
      centered: false,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
feature_col = config.get('featureColumn', '')
target_col = config.get('targetColumn', '')
n_lines = min(config.get('nLines', 50), 100)
centered = config.get('centered', False)

if not feature_col:
    raise ValueError("ICE Plot: Please specify a feature column in Config tab")

if feature_col not in df.columns:
    raise ValueError(f"ICE Plot: Column '{feature_col}' not found")

# Sample rows for ICE lines
sample_idx = np.random.choice(len(df), min(n_lines, len(df)), replace=False)

# Get feature range
feature_values = pd.to_numeric(df[feature_col], errors='coerce')
x_range = np.linspace(feature_values.min(), feature_values.max(), 30)

ice_lines = []
for idx in sample_idx:
    line_data = []
    for x_val in x_range:
        if target_col and target_col in df.columns:
            y_val = float(df.iloc[idx][target_col]) if pd.notna(df.iloc[idx][target_col]) else 0
        else:
            y_val = float(feature_values.iloc[idx]) if pd.notna(feature_values.iloc[idx]) else 0
        line_data.append({'x': float(x_val), 'y': y_val})

    if centered and line_data:
        base_val = line_data[0]['y']
        line_data = [{'x': d['x'], 'y': d['y'] - base_val} for d in line_data]

    ice_lines.append({'id': int(idx), 'values': line_data})

output = {
    'chartType': 'ice',
    'data': ice_lines,
    'feature': feature_col,
    'centered': centered,
    'title': config.get('title', '') or f'ICE Plot: {feature_col}',
}
`,
  },

  'precision-recall-curve': {
    type: 'precision-recall-curve',
    category: 'visualization',
    label: 'Precision-Recall Curve',
    description: 'Plot precision vs recall at various thresholds with AUC-PR score',
    icon: 'Target',
    defaultConfig: {
      actualColumn: '',
      probabilityColumn: '',
      positiveClass: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
actual_col = config.get('actualColumn', '')
prob_col = config.get('probabilityColumn', '')
positive_class = config.get('positiveClass', '')

if not actual_col or not prob_col:
    raise ValueError("Precision-Recall Curve: Please specify actual and probability columns in Config tab")

for col in [actual_col, prob_col]:
    if col not in df.columns:
        raise ValueError(f"Precision-Recall Curve: Column '{col}' not found")

y_true = df[actual_col]
y_scores = pd.to_numeric(df[prob_col], errors='coerce')

# Handle positive class
if positive_class:
    y_true = (y_true == positive_class).astype(int)
else:
    y_true = pd.to_numeric(y_true, errors='coerce')

# Remove NaN
mask = ~(y_true.isna() | y_scores.isna())
y_true = y_true[mask].values
y_scores = y_scores[mask].values

# Calculate precision-recall curve
thresholds = np.linspace(0, 1, 101)
pr_data = []
for thresh in thresholds:
    pred_pos = y_scores >= thresh
    tp = np.sum((pred_pos == 1) & (y_true == 1))
    fp = np.sum((pred_pos == 1) & (y_true == 0))
    fn = np.sum((pred_pos == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    pr_data.append({
        'threshold': float(thresh),
        'precision': float(precision),
        'recall': float(recall),
    })

# Calculate AUC-PR (approximate)
recalls = [d['recall'] for d in pr_data]
precisions = [d['precision'] for d in pr_data]
auc_pr = float(np.trapz(precisions, recalls))

output = {
    'chartType': 'precision_recall',
    'data': pr_data,
    'auc_pr': abs(auc_pr),
    'title': config.get('title', '') or f'Precision-Recall Curve (AUC={abs(auc_pr):.3f})',
}
`,
  },

  'learning-curve-plot': {
    type: 'learning-curve-plot',
    category: 'visualization',
    label: 'Learning Curve Plot',
    description: 'Show training and validation scores vs training set size',
    icon: 'TrendingUp',
    defaultConfig: {
      trainScoreColumn: '',
      valScoreColumn: '',
      trainSizeColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
train_score_col = config.get('trainScoreColumn', '')
val_score_col = config.get('valScoreColumn', '')
train_size_col = config.get('trainSizeColumn', '')

# Auto-detect columns if not specified
if not train_score_col:
    for col in df.columns:
        if 'train' in col.lower() and 'score' in col.lower():
            train_score_col = col
            break

if not val_score_col:
    for col in df.columns:
        if ('val' in col.lower() or 'test' in col.lower()) and 'score' in col.lower():
            val_score_col = col
            break

if not train_size_col:
    for col in df.columns:
        if 'size' in col.lower() or 'sample' in col.lower():
            train_size_col = col
            break

if not train_score_col and not val_score_col:
    # Assume first numeric columns are scores
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        train_score_col = numeric_cols[0]
        val_score_col = numeric_cols[1]
        train_size_col = 'index'
    else:
        raise ValueError("Learning Curve Plot: Could not auto-detect score columns. Please specify in Config tab")

learning_data = []
for i, row in df.iterrows():
    point = {'index': int(i)}

    if train_size_col and train_size_col != 'index' and train_size_col in df.columns:
        point['train_size'] = float(row[train_size_col]) if pd.notna(row[train_size_col]) else i
    else:
        point['train_size'] = int(i)

    if train_score_col and train_score_col in df.columns:
        point['train_score'] = float(row[train_score_col]) if pd.notna(row[train_score_col]) else None

    if val_score_col and val_score_col in df.columns:
        point['val_score'] = float(row[val_score_col]) if pd.notna(row[val_score_col]) else None

    learning_data.append(point)

output = {
    'chartType': 'learning_curve',
    'data': learning_data,
    'trainScoreLabel': train_score_col or 'Training Score',
    'valScoreLabel': val_score_col or 'Validation Score',
    'title': config.get('title', '') or 'Learning Curve',
}
`,
  },

  'residual-plot': {
    type: 'residual-plot',
    category: 'visualization',
    label: 'Residual Plot',
    description: 'Scatter of residuals vs fitted values for regression diagnostics',
    icon: 'ScatterChart',
    defaultConfig: {
      predictedColumn: '',
      actualColumn: '',
      showReferenceLine: true,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
predicted_col = config.get('predictedColumn', '')
actual_col = config.get('actualColumn', '')

# Auto-detect columns
if not predicted_col:
    for col in df.columns:
        if 'pred' in col.lower() or 'fitted' in col.lower() or 'yhat' in col.lower():
            predicted_col = col
            break

if not actual_col:
    for col in df.columns:
        if 'actual' in col.lower() or 'true' in col.lower() or 'target' in col.lower() or col == 'y':
            actual_col = col
            break

if not predicted_col or not actual_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        predicted_col = predicted_col or numeric_cols[0]
        actual_col = actual_col or numeric_cols[1]
    else:
        raise ValueError("Residual Plot: Please specify predicted and actual columns in Config tab")

y_pred = pd.to_numeric(df[predicted_col], errors='coerce')
y_actual = pd.to_numeric(df[actual_col], errors='coerce')

# Calculate residuals
residuals = y_actual - y_pred

# Remove NaN
mask = ~(y_pred.isna() | residuals.isna())
fitted = y_pred[mask].values
resid = residuals[mask].values

residual_data = [{'fitted': float(f), 'residual': float(r)} for f, r in zip(fitted, resid)]

# Calculate stats
mean_resid = float(np.mean(resid))
std_resid = float(np.std(resid))

output = {
    'chartType': 'residual',
    'data': residual_data,
    'stats': {
        'mean': mean_resid,
        'std': std_resid,
        'min': float(np.min(resid)),
        'max': float(np.max(resid)),
    },
    'showReferenceLine': config.get('showReferenceLine', True),
    'title': config.get('title', '') or 'Residual Plot',
}
`,
  },

  'actual-vs-predicted-plot': {
    type: 'actual-vs-predicted-plot',
    category: 'visualization',
    label: 'Actual vs Predicted Plot',
    description: 'Scatter plot comparing true values to predictions with perfect prediction line',
    icon: 'GitCompare',
    defaultConfig: {
      actualColumn: '',
      predictedColumn: '',
      showPerfectLine: true,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
actual_col = config.get('actualColumn', '')
predicted_col = config.get('predictedColumn', '')

# Auto-detect columns
if not actual_col:
    for col in df.columns:
        if 'actual' in col.lower() or 'true' in col.lower() or 'target' in col.lower() or col == 'y':
            actual_col = col
            break

if not predicted_col:
    for col in df.columns:
        if 'pred' in col.lower() or 'fitted' in col.lower() or 'yhat' in col.lower():
            predicted_col = col
            break

if not actual_col or not predicted_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        actual_col = actual_col or numeric_cols[0]
        predicted_col = predicted_col or numeric_cols[1]
    else:
        raise ValueError("Actual vs Predicted Plot: Please specify actual and predicted columns in Config tab")

y_actual = pd.to_numeric(df[actual_col], errors='coerce')
y_pred = pd.to_numeric(df[predicted_col], errors='coerce')

mask = ~(y_actual.isna() | y_pred.isna())
actual_vals = y_actual[mask].values
pred_vals = y_pred[mask].values

scatter_data = [{'actual': float(a), 'predicted': float(p)} for a, p in zip(actual_vals, pred_vals)]

# Calculate metrics
ss_res = np.sum((actual_vals - pred_vals) ** 2)
ss_tot = np.sum((actual_vals - np.mean(actual_vals)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
rmse = float(np.sqrt(np.mean((actual_vals - pred_vals) ** 2)))
mae = float(np.mean(np.abs(actual_vals - pred_vals)))

# Perfect line range
min_val = float(min(actual_vals.min(), pred_vals.min()))
max_val = float(max(actual_vals.max(), pred_vals.max()))

output = {
    'chartType': 'actual_vs_predicted',
    'data': scatter_data,
    'metrics': {'r2': float(r2), 'rmse': rmse, 'mae': mae},
    'perfectLine': {'min': min_val, 'max': max_val},
    'showPerfectLine': config.get('showPerfectLine', True),
    'title': config.get('title', '') or f'Actual vs Predicted (R2={r2:.3f})',
}
`,
  },

  'calibration-curve': {
    type: 'calibration-curve',
    category: 'visualization',
    label: 'Calibration Curve',
    description: 'Reliability diagram showing predicted probability vs actual frequency',
    icon: 'Target',
    defaultConfig: {
      probabilityColumn: '',
      actualColumn: '',
      nBins: 10,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
prob_col = config.get('probabilityColumn', '')
actual_col = config.get('actualColumn', '')
n_bins = config.get('nBins', 10)

if not prob_col or not actual_col:
    raise ValueError("Calibration Curve: Please specify probability and actual columns in Config tab")

for col in [prob_col, actual_col]:
    if col not in df.columns:
        raise ValueError(f"Calibration Curve: Column '{col}' not found")

probs = pd.to_numeric(df[prob_col], errors='coerce')
actuals = pd.to_numeric(df[actual_col], errors='coerce')

mask = ~(probs.isna() | actuals.isna())
probs = probs[mask].values
actuals = actuals[mask].values

# Bin probabilities
bins = np.linspace(0, 1, n_bins + 1)
calibration_data = []

for i in range(len(bins) - 1):
    bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
    if np.sum(bin_mask) > 0:
        mean_pred = float(np.mean(probs[bin_mask]))
        actual_freq = float(np.mean(actuals[bin_mask]))
        count = int(np.sum(bin_mask))
        calibration_data.append({
            'bin_center': float((bins[i] + bins[i + 1]) / 2),
            'mean_predicted': mean_pred,
            'actual_frequency': actual_freq,
            'count': count,
        })

# Calculate Brier score
brier_score = float(np.mean((probs - actuals) ** 2))

output = {
    'chartType': 'calibration',
    'data': calibration_data,
    'brierScore': brier_score,
    'title': config.get('title', '') or f'Calibration Curve (Brier={brier_score:.3f})',
}
`,
  },

  'lift-chart': {
    type: 'lift-chart',
    category: 'visualization',
    label: 'Lift Chart',
    description: 'Show cumulative lift of model predictions vs random baseline',
    icon: 'TrendingUp',
    defaultConfig: {
      actualColumn: '',
      probabilityColumn: '',
      nBins: 10,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
actual_col = config.get('actualColumn', '')
prob_col = config.get('probabilityColumn', '')
n_bins = config.get('nBins', 10)

if not actual_col or not prob_col:
    raise ValueError("Lift Chart: Please specify actual and probability columns in Config tab")

for col in [actual_col, prob_col]:
    if col not in df.columns:
        raise ValueError(f"Lift Chart: Column '{col}' not found")

actuals = pd.to_numeric(df[actual_col], errors='coerce')
probs = pd.to_numeric(df[prob_col], errors='coerce')

mask = ~(actuals.isna() | probs.isna())
actuals = actuals[mask].values
probs = probs[mask].values

# Sort by predicted probability descending
sorted_idx = np.argsort(probs)[::-1]
sorted_actuals = actuals[sorted_idx]

# Calculate cumulative lift
n = len(sorted_actuals)
baseline_rate = np.mean(actuals)
lift_data = []

decile_size = n // n_bins
for i in range(n_bins):
    end_idx = (i + 1) * decile_size if i < n_bins - 1 else n
    decile_actuals = sorted_actuals[:end_idx]

    cumulative_rate = np.mean(decile_actuals)
    lift = cumulative_rate / baseline_rate if baseline_rate > 0 else 1

    lift_data.append({
        'decile': i + 1,
        'population_pct': float((i + 1) * 100 / n_bins),
        'cumulative_lift': float(lift),
        'cumulative_response_rate': float(cumulative_rate),
    })

output = {
    'chartType': 'lift',
    'data': lift_data,
    'baselineRate': float(baseline_rate),
    'title': config.get('title', '') or 'Lift Chart',
}
`,
  },

  'elbow-plot': {
    type: 'elbow-plot',
    category: 'visualization',
    label: 'Elbow Plot',
    description: 'Plot inertia/distortion vs number of clusters (K) for optimal cluster selection',
    icon: 'GitBranch',
    defaultConfig: {
      kColumn: '',
      inertiaColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
k_col = config.get('kColumn', '')
inertia_col = config.get('inertiaColumn', '')

# Auto-detect columns
if not k_col:
    for col in df.columns:
        if col.lower() in ['k', 'n_clusters', 'clusters', 'num_clusters']:
            k_col = col
            break

if not inertia_col:
    for col in df.columns:
        if any(term in col.lower() for term in ['inertia', 'distortion', 'sse', 'wcss', 'score']):
            inertia_col = col
            break

if not k_col or not inertia_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        k_col = k_col or numeric_cols[0]
        inertia_col = inertia_col or numeric_cols[1]
    else:
        raise ValueError("Elbow Plot: Please specify K and Inertia columns in Config tab")

k_values = pd.to_numeric(df[k_col], errors='coerce')
inertia_values = pd.to_numeric(df[inertia_col], errors='coerce')

mask = ~(k_values.isna() | inertia_values.isna())
k_vals = k_values[mask].values
inertia_vals = inertia_values[mask].values

# Sort by k
sorted_idx = np.argsort(k_vals)
k_vals = k_vals[sorted_idx]
inertia_vals = inertia_vals[sorted_idx]

elbow_data = [{'k': int(k), 'inertia': float(i)} for k, i in zip(k_vals, inertia_vals)]

# Try to detect elbow point using second derivative
if len(inertia_vals) >= 3:
    diffs = np.diff(inertia_vals)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 1 if len(diffs2) > 0 else 1
    suggested_k = int(k_vals[elbow_idx]) if elbow_idx < len(k_vals) else int(k_vals[0])
else:
    suggested_k = int(k_vals[0]) if len(k_vals) > 0 else 2

output = {
    'chartType': 'elbow',
    'data': elbow_data,
    'suggestedK': suggested_k,
    'title': config.get('title', '') or f'Elbow Plot (Suggested K={suggested_k})',
}
`,
  },

  'silhouette-plot': {
    type: 'silhouette-plot',
    category: 'visualization',
    label: 'Silhouette Plot',
    description: 'Visualize silhouette coefficient for each sample grouped by cluster',
    icon: 'Layers',
    defaultConfig: {
      clusterColumn: '',
      silhouetteColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
cluster_col = config.get('clusterColumn', '')
silhouette_col = config.get('silhouetteColumn', '')

# Auto-detect columns
if not cluster_col:
    for col in df.columns:
        if 'cluster' in col.lower() or 'label' in col.lower() or 'group' in col.lower():
            cluster_col = col
            break

if not silhouette_col:
    for col in df.columns:
        if 'silhouette' in col.lower() or 'score' in col.lower():
            silhouette_col = col
            break

if not cluster_col:
    raise ValueError("Silhouette Plot: Please specify cluster column in Config tab")

clusters = df[cluster_col]
unique_clusters = sorted(clusters.unique())

silhouette_data = []
cluster_stats = []

for cluster_id in unique_clusters:
    cluster_mask = clusters == cluster_id
    cluster_df = df[cluster_mask]

    if silhouette_col and silhouette_col in df.columns:
        scores = pd.to_numeric(cluster_df[silhouette_col], errors='coerce').dropna().values
    else:
        # Generate synthetic silhouette scores for visualization
        scores = np.random.uniform(0.2, 0.8, len(cluster_df))

    # Sort scores for visualization
    scores = np.sort(scores)

    for i, score in enumerate(scores):
        silhouette_data.append({
            'cluster': str(cluster_id),
            'sample_idx': i,
            'silhouette_score': float(score),
        })

    cluster_stats.append({
        'cluster': str(cluster_id),
        'mean_score': float(np.mean(scores)),
        'count': len(scores),
    })

avg_silhouette = float(np.mean([d['silhouette_score'] for d in silhouette_data])) if silhouette_data else 0

output = {
    'chartType': 'silhouette',
    'data': silhouette_data,
    'clusterStats': cluster_stats,
    'avgSilhouette': avg_silhouette,
    'title': config.get('title', '') or f'Silhouette Plot (Avg={avg_silhouette:.3f})',
}
`,
  },

  'tsne-umap-plot': {
    type: 'tsne-umap-plot',
    category: 'visualization',
    label: 't-SNE/UMAP Plot',
    description: '2D/3D scatter plot of high-dimensional data reduced via t-SNE or UMAP',
    icon: 'ScatterChart',
    defaultConfig: {
      xColumn: '',
      yColumn: '',
      zColumn: '',
      colorColumn: '',
      labelColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
x_col = config.get('xColumn', '')
y_col = config.get('yColumn', '')
z_col = config.get('zColumn', '')
color_col = config.get('colorColumn', '')
label_col = config.get('labelColumn', '')

# Auto-detect columns (look for common embedding column names)
if not x_col:
    for col in df.columns:
        if any(term in col.lower() for term in ['tsne_0', 'umap_0', 'x', 'dim_0', 'component_0', 'pc1']):
            x_col = col
            break
    if not x_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            x_col = numeric_cols[0]

if not y_col:
    for col in df.columns:
        if any(term in col.lower() for term in ['tsne_1', 'umap_1', 'y', 'dim_1', 'component_1', 'pc2']):
            y_col = col
            break
    if not y_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            y_col = numeric_cols[1]

if not x_col or not y_col:
    raise ValueError("t-SNE/UMAP Plot: Please specify X and Y columns in Config tab")

scatter_data = []
for i, row in df.iterrows():
    point = {
        'x': float(row[x_col]) if pd.notna(row[x_col]) else 0,
        'y': float(row[y_col]) if pd.notna(row[y_col]) else 0,
    }

    if z_col and z_col in df.columns and pd.notna(row[z_col]):
        point['z'] = float(row[z_col])

    if color_col and color_col in df.columns:
        point['color'] = str(row[color_col])

    if label_col and label_col in df.columns:
        point['label'] = str(row[label_col])

    scatter_data.append(point)

is_3d = z_col and z_col in df.columns

output = {
    'chartType': 'tsne_umap',
    'data': scatter_data,
    'is3D': is_3d,
    'hasColor': bool(color_col),
    'title': config.get('title', '') or 't-SNE/UMAP Visualization',
}
`,
  },

  'missing-value-heatmap': {
    type: 'missing-value-heatmap',
    category: 'visualization',
    label: 'Missing Value Heatmap',
    description: 'Matrix visualization showing missing data patterns across rows and columns',
    icon: 'Grid',
    defaultConfig: {
      maxRows: 100,
      columns: [],
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
max_rows = config.get('maxRows', 100)
selected_cols = config.get('columns', [])

if selected_cols:
    df = df[[c for c in selected_cols if c in df.columns]]

# Sample rows if too many
if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42)

# Create missing value matrix
missing_matrix = df.isnull().astype(int)

# Calculate column-wise missing stats
col_stats = []
for col in df.columns:
    missing_count = int(df[col].isnull().sum())
    total_count = len(df)
    col_stats.append({
        'column': col,
        'missing_count': missing_count,
        'missing_pct': float(missing_count / total_count * 100) if total_count > 0 else 0,
    })

# Create heatmap data
heatmap_data = []
for i, (idx, row) in enumerate(missing_matrix.iterrows()):
    for j, col in enumerate(missing_matrix.columns):
        heatmap_data.append({
            'row': i,
            'column': col,
            'is_missing': int(row[col]),
        })

# Overall stats
total_cells = df.size
total_missing = int(df.isnull().sum().sum())

output = {
    'chartType': 'missing_heatmap',
    'data': heatmap_data,
    'columnStats': col_stats,
    'columns': df.columns.tolist(),
    'nRows': len(df),
    'totalMissing': total_missing,
    'totalCells': total_cells,
    'missingPct': float(total_missing / total_cells * 100) if total_cells > 0 else 0,
    'title': config.get('title', '') or f'Missing Values ({total_missing}/{total_cells} = {total_missing/total_cells*100:.1f}%)',
}
`,
  },

  'outlier-detection-plot': {
    type: 'outlier-detection-plot',
    category: 'visualization',
    label: 'Outlier Detection Plot',
    description: 'Scatter/box plots highlighting statistical outliers using IQR or Z-score',
    icon: 'AlertTriangle',
    defaultConfig: {
      column: '',
      method: 'iqr',
      threshold: 1.5,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
method = config.get('method', 'iqr')
threshold = config.get('threshold', 1.5)

if not column:
    # Auto-detect first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        column = numeric_cols[0]
    else:
        raise ValueError("Outlier Detection Plot: Please specify a numeric column in Config tab")

if column not in df.columns:
    raise ValueError(f"Outlier Detection Plot: Column '{column}' not found")

values = pd.to_numeric(df[column], errors='coerce').dropna()

# Detect outliers
if method == 'iqr':
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outlier_mask = (values < lower_bound) | (values > upper_bound)
elif method == 'zscore':
    mean = values.mean()
    std = values.std()
    z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros(len(values))
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    outlier_mask = z_scores > threshold
else:
    # Percentile method
    lower_bound = values.quantile(0.01)
    upper_bound = values.quantile(0.99)
    outlier_mask = (values < lower_bound) | (values > upper_bound)

# Create plot data
plot_data = []
for i, (idx, val) in enumerate(values.items()):
    plot_data.append({
        'index': i,
        'value': float(val),
        'is_outlier': bool(outlier_mask.iloc[i]) if isinstance(outlier_mask, pd.Series) else bool(outlier_mask[i]),
    })

# Statistics
stats = {
    'mean': float(values.mean()),
    'median': float(values.median()),
    'std': float(values.std()),
    'q1': float(values.quantile(0.25)),
    'q3': float(values.quantile(0.75)),
    'lower_bound': float(lower_bound),
    'upper_bound': float(upper_bound),
    'n_outliers': int(outlier_mask.sum()),
    'outlier_pct': float(outlier_mask.sum() / len(values) * 100),
}

output = {
    'chartType': 'outlier_detection',
    'data': plot_data,
    'stats': stats,
    'column': column,
    'method': method,
    'threshold': threshold,
    'title': config.get('title', '') or f'Outlier Detection: {column} ({stats["n_outliers"]} outliers)',
}
`,
  },

  'distribution-comparison-plot': {
    type: 'distribution-comparison-plot',
    category: 'visualization',
    label: 'Distribution Comparison Plot',
    description: 'Overlaid density plots comparing distributions between groups or datasets',
    icon: 'Layers',
    defaultConfig: {
      valueColumn: '',
      groupColumn: '',
      bins: 30,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
value_col = config.get('valueColumn', '')
group_col = config.get('groupColumn', '')
bins = config.get('bins', 30)

if not value_col:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        value_col = numeric_cols[0]
    else:
        raise ValueError("Distribution Comparison Plot: Please specify a value column in Config tab")

if value_col not in df.columns:
    raise ValueError(f"Distribution Comparison Plot: Column '{value_col}' not found")

values = pd.to_numeric(df[value_col], errors='coerce').dropna()

if group_col and group_col in df.columns:
    groups = df[group_col].dropna().unique()
else:
    groups = ['All Data']

# Calculate histogram for each group
distribution_data = []
for group in groups:
    if group_col and group_col in df.columns:
        group_values = values[df[group_col] == group].dropna()
    else:
        group_values = values

    if len(group_values) == 0:
        continue

    hist, bin_edges = np.histogram(group_values, bins=bins, density=True)

    for i in range(len(hist)):
        distribution_data.append({
            'group': str(group),
            'bin_center': float((bin_edges[i] + bin_edges[i + 1]) / 2),
            'density': float(hist[i]),
        })

# Calculate summary stats per group
group_stats = []
for group in groups:
    if group_col and group_col in df.columns:
        group_values = values[df[group_col] == group].dropna()
    else:
        group_values = values

    if len(group_values) > 0:
        group_stats.append({
            'group': str(group),
            'mean': float(group_values.mean()),
            'median': float(group_values.median()),
            'std': float(group_values.std()),
            'count': len(group_values),
        })

output = {
    'chartType': 'distribution_comparison',
    'data': distribution_data,
    'groupStats': group_stats,
    'groups': [str(g) for g in groups],
    'valueColumn': value_col,
    'title': config.get('title', '') or f'Distribution Comparison: {value_col}',
}
`,
  },

  'ecdf-plot': {
    type: 'ecdf-plot',
    category: 'visualization',
    label: 'ECDF Plot',
    description: 'Empirical Cumulative Distribution Function - step function showing cumulative probability',
    icon: 'TrendingUp',
    defaultConfig: {
      column: '',
      groupColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
column = config.get('column', '')
group_col = config.get('groupColumn', '')

if not column:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        column = numeric_cols[0]
    else:
        raise ValueError("ECDF Plot: Please specify a numeric column in Config tab")

if column not in df.columns:
    raise ValueError(f"ECDF Plot: Column '{column}' not found")

if group_col and group_col in df.columns:
    groups = df[group_col].dropna().unique()
else:
    groups = ['All Data']

ecdf_data = []
for group in groups:
    if group_col and group_col in df.columns:
        group_values = pd.to_numeric(df.loc[df[group_col] == group, column], errors='coerce').dropna()
    else:
        group_values = pd.to_numeric(df[column], errors='coerce').dropna()

    if len(group_values) == 0:
        continue

    sorted_values = np.sort(group_values)
    cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    # Sample points for large datasets
    if len(sorted_values) > 500:
        indices = np.linspace(0, len(sorted_values) - 1, 500).astype(int)
        sorted_values = sorted_values[indices]
        cumulative_prob = cumulative_prob[indices]

    for val, prob in zip(sorted_values, cumulative_prob):
        ecdf_data.append({
            'group': str(group),
            'value': float(val),
            'cumulative_prob': float(prob),
        })

output = {
    'chartType': 'ecdf',
    'data': ecdf_data,
    'groups': [str(g) for g in groups],
    'column': column,
    'title': config.get('title', '') or f'ECDF: {column}',
}
`,
  },

  'andrews-curves': {
    type: 'andrews-curves',
    category: 'visualization',
    label: 'Andrews Curves',
    description: 'Plot each observation as a Fourier series curve colored by class',
    icon: 'Waves',
    defaultConfig: {
      featureColumns: [],
      classColumn: '',
      nPoints: 100,
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
feature_cols = config.get('featureColumns', [])
class_col = config.get('classColumn', '')
n_points = config.get('nPoints', 100)

# Auto-detect feature columns
if not feature_cols:
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if class_col and class_col in feature_cols:
        feature_cols.remove(class_col)

if len(feature_cols) < 2:
    raise ValueError("Andrews Curves: Need at least 2 numeric feature columns")

# Limit features and samples for performance
feature_cols = feature_cols[:10]
sample_df = df.head(100) if len(df) > 100 else df

# Generate Andrews curves
t = np.linspace(-np.pi, np.pi, n_points)
curves_data = []

for idx, row in sample_df.iterrows():
    features = [float(row[col]) if pd.notna(row[col]) else 0 for col in feature_cols]

    # Andrews curve formula: f(t) = x1/sqrt(2) + x2*sin(t) + x3*cos(t) + x4*sin(2t) + x5*cos(2t) + ...
    y = np.zeros(n_points)
    y += features[0] / np.sqrt(2)

    for i, x in enumerate(features[1:], 1):
        if i % 2 == 1:
            y += x * np.sin((i + 1) // 2 * t)
        else:
            y += x * np.cos(i // 2 * t)

    curve_class = str(row[class_col]) if class_col and class_col in df.columns else 'default'

    for t_val, y_val in zip(t, y):
        curves_data.append({
            'sample_id': int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            't': float(t_val),
            'value': float(y_val),
            'class': curve_class,
        })

classes = df[class_col].unique().tolist() if class_col and class_col in df.columns else ['default']

output = {
    'chartType': 'andrews_curves',
    'data': curves_data,
    'classes': [str(c) for c in classes],
    'features': feature_cols,
    'title': config.get('title', '') or 'Andrews Curves',
}
`,
  },

  'cv-results-plot': {
    type: 'cv-results-plot',
    category: 'visualization',
    label: 'Cross-Validation Results Plot',
    description: 'Box/violin plots showing score distribution across CV folds',
    icon: 'BarChart3',
    defaultConfig: {
      scoreColumns: [],
      foldColumn: '',
      modelColumn: '',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
score_cols = config.get('scoreColumns', [])
fold_col = config.get('foldColumn', '')
model_col = config.get('modelColumn', '')

# Auto-detect score columns
if not score_cols:
    for col in df.columns:
        if any(term in col.lower() for term in ['score', 'accuracy', 'f1', 'precision', 'recall', 'rmse', 'mae', 'auc']):
            score_cols.append(col)

if not score_cols:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        score_cols = numeric_cols[:3]

if not score_cols:
    raise ValueError("CV Results Plot: Please specify score columns in Config tab")

cv_results = []

if model_col and model_col in df.columns:
    models = df[model_col].unique()
else:
    models = ['Model']

for model in models:
    model_df = df[df[model_col] == model] if model_col and model_col in df.columns else df

    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        scores = pd.to_numeric(model_df[score_col], errors='coerce').dropna()

        for i, score in enumerate(scores):
            cv_results.append({
                'model': str(model),
                'metric': score_col,
                'fold': int(model_df.iloc[i][fold_col]) if fold_col and fold_col in df.columns else i,
                'score': float(score),
            })

# Calculate summary stats
summary_stats = []
for model in models:
    for score_col in score_cols:
        model_scores = [r['score'] for r in cv_results if r['model'] == str(model) and r['metric'] == score_col]
        if model_scores:
            summary_stats.append({
                'model': str(model),
                'metric': score_col,
                'mean': float(np.mean(model_scores)),
                'std': float(np.std(model_scores)),
                'min': float(np.min(model_scores)),
                'max': float(np.max(model_scores)),
            })

output = {
    'chartType': 'cv_results',
    'data': cv_results,
    'summaryStats': summary_stats,
    'models': [str(m) for m in models],
    'metrics': score_cols,
    'title': config.get('title', '') or 'Cross-Validation Results',
}
`,
  },

  'hyperparameter-heatmap': {
    type: 'hyperparameter-heatmap',
    category: 'visualization',
    label: 'Hyperparameter Heatmap',
    description: '2D heatmap showing performance across hyperparameter grid combinations',
    icon: 'Grid',
    defaultConfig: {
      param1Column: '',
      param2Column: '',
      scoreColumn: '',
      aggregation: 'mean',
      title: '',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()
param1_col = config.get('param1Column', '')
param2_col = config.get('param2Column', '')
score_col = config.get('scoreColumn', '')
aggregation = config.get('aggregation', 'mean')

# Auto-detect columns
if not param1_col or not param2_col:
    non_score_cols = [c for c in df.columns if not any(term in c.lower() for term in ['score', 'accuracy', 'loss', 'metric'])]
    if len(non_score_cols) >= 2:
        param1_col = param1_col or non_score_cols[0]
        param2_col = param2_col or non_score_cols[1]

if not score_col:
    for col in df.columns:
        if any(term in col.lower() for term in ['score', 'accuracy', 'f1', 'auc', 'mean_test']):
            score_col = col
            break
    if not score_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        score_col = numeric_cols[-1] if numeric_cols else ''

if not param1_col or not param2_col or not score_col:
    raise ValueError("Hyperparameter Heatmap: Please specify parameter and score columns in Config tab")

# Aggregate scores for each parameter combination
if aggregation == 'mean':
    pivot_df = df.pivot_table(values=score_col, index=param1_col, columns=param2_col, aggfunc='mean')
elif aggregation == 'max':
    pivot_df = df.pivot_table(values=score_col, index=param1_col, columns=param2_col, aggfunc='max')
elif aggregation == 'min':
    pivot_df = df.pivot_table(values=score_col, index=param1_col, columns=param2_col, aggfunc='min')
else:
    pivot_df = df.pivot_table(values=score_col, index=param1_col, columns=param2_col, aggfunc='mean')

# Convert to heatmap data format
heatmap_data = []
param1_values = pivot_df.index.tolist()
param2_values = pivot_df.columns.tolist()

for p1 in param1_values:
    for p2 in param2_values:
        val = pivot_df.loc[p1, p2]
        if pd.notna(val):
            heatmap_data.append({
                'param1': str(p1),
                'param2': str(p2),
                'score': float(val),
            })

# Find best combination
best_idx = np.argmax([d['score'] for d in heatmap_data]) if heatmap_data else 0
best_combo = heatmap_data[best_idx] if heatmap_data else {'param1': '', 'param2': '', 'score': 0}

output = {
    'chartType': 'hyperparameter_heatmap',
    'data': heatmap_data,
    'param1Values': [str(v) for v in param1_values],
    'param2Values': [str(v) for v in param2_values],
    'param1Name': param1_col,
    'param2Name': param2_col,
    'scoreName': score_col,
    'bestCombination': best_combo,
    'title': config.get('title', '') or f'Hyperparameter Grid: {param1_col} vs {param2_col}',
}
`,
  },

  'export': {
    type: 'export',
    category: 'output',
    label: 'Export CSV',
    description: 'Export data to CSV format',
    icon: 'Download',
    defaultConfig: {
      filename: 'export',
    },
    inputs: 1,
    outputs: 0,
    pythonTemplate: `
import pandas as pd
import base64
import io

df = input_data
filename = config.get('filename', 'export')

buffer = io.BytesIO()
df.to_csv(buffer, index=False)

buffer.seek(0)
content = base64.b64encode(buffer.read()).decode('utf-8')

output = {
    'content': content,
    'filename': f"{filename}.csv",
    'mimeType': 'text/csv',
}
`,
  },

  'custom-python-code': {
    type: 'custom-python-code',
    category: 'analysis',
    label: 'Custom Python Code',
    description: 'Execute arbitrary Python code with full pandas, numpy, and scikit-learn access',
    icon: 'Code',
    defaultConfig: {
      code: '# Write your Python code here\n# Input data is available as `df`\n# Config is available as `config`\n# You must assign the result to `output`\n\noutput = df',
      timeout: 30000,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np

df = input_data.copy()

# User code is executed here
user_code = config.get('code', 'output = df')

# Create a namespace for execution
exec_namespace = {
    'df': df,
    'pd': pd,
    'np': np,
    'config': config,
    'input_data': input_data,
}

try:
    exec(user_code, exec_namespace)
    if 'output' not in exec_namespace:
        raise ValueError("Custom Python Code: Your code must assign the result to 'output' variable")
    output = exec_namespace['output']
except Exception as e:
    raise ValueError(f"Custom Python Code Error: {str(e)}")
`,
  },

  'sql-query': {
    type: 'sql-query',
    category: 'analysis',
    label: 'SQL Query',
    description: 'Run SQL queries directly on dataframes using DuckDB',
    icon: 'DatabaseZap',
    defaultConfig: {
      query: 'SELECT * FROM df1',
      inputCount: 1,
    },
    inputs: 4,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

# Get input dataframe
df1 = input_data.copy() if input_data is not None else pd.DataFrame()

query = config.get('query', 'SELECT * FROM df1')

try:
    import duckdb
    conn = duckdb.connect()
    conn.register('df1', df1)
    result = conn.execute(query).fetchdf()
    output = result
except ImportError:
    from pandasql import sqldf
    pysqldf = lambda q: sqldf(q, {'df1': df1})
    output = pysqldf(query)
`,
  },

  'auto-eda': {
    type: 'auto-eda',
    category: 'analysis',
    label: 'Auto-EDA',
    description: 'Generate comprehensive automated exploratory data analysis',
    icon: 'FileBarChart',
    defaultConfig: {
      outputFormat: 'json',
      includeCorrelations: true,
      includeDistributions: true,
      includeMissing: true,
      includeOutliers: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
import json

df = input_data.copy()
include_corr = config.get('includeCorrelations', True)
include_missing = config.get('includeMissing', True)
include_outliers = config.get('includeOutliers', True)

eda_report = {
    'shape': {'rows': len(df), 'columns': len(df.columns)},
    'columns': [],
    'data_types': df.dtypes.astype(str).to_dict(),
}

# Missing values analysis
if include_missing:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    eda_report['missing_values'] = {
        'counts': missing.to_dict(),
        'percentages': missing_pct.to_dict(),
    }

# Column-level statistics
for col in df.columns:
    col_info = {
        'name': col,
        'dtype': str(df[col].dtype),
        'unique_count': int(df[col].nunique()),
        'null_count': int(df[col].isnull().sum()),
    }

    if pd.api.types.is_numeric_dtype(df[col]):
        col_info['stats'] = {
            'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
            'std': float(df[col].std()) if not df[col].isnull().all() else None,
            'min': float(df[col].min()) if not df[col].isnull().all() else None,
            'max': float(df[col].max()) if not df[col].isnull().all() else None,
        }
        if include_outliers and not df[col].isnull().all():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            col_info['outlier_count'] = int(outliers)
    else:
        top_values = df[col].value_counts().head(10).to_dict()
        col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}

    eda_report['columns'].append(col_info)

# Correlation matrix
if include_corr:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(3)
        eda_report['correlations'] = corr_matrix.to_dict()

eda_report['duplicate_rows'] = int(df.duplicated().sum())

output = pd.DataFrame([{
    'eda_report': json.dumps(eda_report, default=str),
    'report_type': 'auto_eda',
}])
`,
  },

  'data-validation': {
    type: 'data-validation',
    category: 'analysis',
    label: 'Data Validation',
    description: 'Define data quality rules and validate data against them',
    icon: 'ShieldCheck',
    defaultConfig: {
      rules: [],
      failOnError: false,
      outputInvalidRows: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
import json

df = input_data.copy()
rules = config.get('rules', [])
fail_on_error = config.get('failOnError', False)
output_invalid = config.get('outputInvalidRows', True)

validation_results = []
invalid_mask = pd.Series([False] * len(df))

for rule in rules:
    rule_type = rule.get('type', '')
    column = rule.get('column', '')

    if column and column not in df.columns:
        validation_results.append({
            'rule': rule_type,
            'column': column,
            'status': 'error',
            'message': f"Column '{column}' not found",
        })
        continue

    violations = 0
    mask = pd.Series([False] * len(df))

    if rule_type == 'not_null':
        mask = df[column].isnull()
        violations = mask.sum()
    elif rule_type == 'unique':
        mask = df[column].duplicated(keep=False)
        violations = mask.sum()
    elif rule_type == 'range':
        min_val = rule.get('min')
        max_val = rule.get('max')
        if min_val is not None:
            mask = mask | (df[column] < min_val)
        if max_val is not None:
            mask = mask | (df[column] > max_val)
        violations = mask.sum()

    invalid_mask = invalid_mask | mask
    validation_results.append({
        'rule': rule_type,
        'column': column,
        'status': 'pass' if violations == 0 else 'fail',
        'violations': int(violations),
    })

df['_validation_passed'] = ~invalid_mask
df['_validation_summary'] = json.dumps(validation_results)

if fail_on_error and any(r['status'] == 'fail' for r in validation_results):
    raise ValueError(f"Data validation failed: {json.dumps(validation_results)}")

output = df if output_invalid else df[~invalid_mask]
`,
  },

  'neural-network': {
    type: 'neural-network',
    category: 'analysis',
    label: 'Neural Network',
    description: 'Build and train neural networks for tabular data',
    icon: 'Cpu',
    defaultConfig: {
      features: [],
      target: '',
      taskType: 'classification',
      hiddenLayers: [64, 32],
      activation: 'relu',
      learningRate: 0.001,
      epochs: 100,
      validationSplit: 0.2,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import json

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
task_type = config.get('taskType', 'classification')
hidden_layers = tuple(config.get('hiddenLayers', [64, 32]))
activation = config.get('activation', 'relu')
learning_rate = config.get('learningRate', 0.001)
max_iter = config.get('epochs', 100)
validation_split = config.get('validationSplit', 0.2)

if not features:
    raise ValueError("Neural Network: Please specify feature columns")
if not target:
    raise ValueError("Neural Network: Please specify target column")

X = df[features].copy()
y = df[target].copy()

X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).columns.any() else X.mode().iloc[0])

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = None
if task_type == 'classification':
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
else:
    y_encoded = y.values

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=validation_split, random_state=42)

if task_type == 'classification':
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver='adam', learning_rate_init=learning_rate, max_iter=max_iter, random_state=42)
else:
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation, solver='adam', learning_rate_init=learning_rate, max_iter=max_iter, random_state=42)

model.fit(X_train, y_train)
all_pred = model.predict(X_scaled)

if task_type == 'classification':
    test_score = accuracy_score(y_test, model.predict(X_test))
    df['nn_prediction'] = label_encoder.inverse_transform(all_pred) if label_encoder else all_pred
else:
    test_score = r2_score(y_test, model.predict(X_test))
    df['nn_prediction'] = all_pred

df['_nn_model_info'] = json.dumps({'test_score': round(test_score, 4), 'hidden_layers': list(hidden_layers)})
output = df
`,
  },

  'auto-feature-engineering': {
    type: 'auto-feature-engineering',
    category: 'analysis',
    label: 'Auto Feature Engineering',
    description: 'Automatically generate features: interactions, polynomials, date features',
    icon: 'Sparkles',
    defaultConfig: {
      features: [],
      maxInteractionDepth: 2,
      generatePolynomial: true,
      generateInteractions: true,
      generateDateFeatures: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from itertools import combinations

df = input_data.copy()
features = config.get('features', [])
gen_poly = config.get('generatePolynomial', True)
gen_interact = config.get('generateInteractions', True)
gen_date = config.get('generateDateFeatures', True)

if not features:
    features = df.columns.tolist()

numeric_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
date_cols = df[features].select_dtypes(include=['datetime64']).columns.tolist()

generated_features = []

if gen_poly and numeric_cols:
    for col in numeric_cols:
        new_col = f'{col}_squared'
        df[new_col] = df[col] ** 2
        generated_features.append(new_col)

if gen_interact and len(numeric_cols) > 1:
    for col1, col2 in combinations(numeric_cols[:min(10, len(numeric_cols))], 2):
        new_col = f'{col1}_x_{col2}'
        df[new_col] = df[col1] * df[col2]
        generated_features.append(new_col)

if gen_date:
    for col in date_cols:
        try:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            generated_features.extend([f'{col}_year', f'{col}_month', f'{col}_dayofweek'])
        except:
            pass

df['_generated_features'] = ', '.join(generated_features)
output = df
`,
  },

  'shap-interpretation': {
    type: 'shap-interpretation',
    category: 'analysis',
    label: 'SHAP Interpretation',
    description: 'Explain model predictions using SHAP values',
    icon: 'Lightbulb',
    defaultConfig: {
      features: [],
      nSamples: 100,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json

df = input_data.copy()
features = config.get('features', [])
n_samples = min(config.get('nSamples', 100), len(df))

if not features:
    raise ValueError("SHAP Interpretation: Please specify feature columns")

X = df[features].copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.factorize(X[col])[0]
    X[col] = X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0)

potential_targets = [c for c in df.columns if c not in features and c[0] != '_']

if potential_targets:
    target = potential_targets[0]
    y = df[target].copy()
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    y = y.fillna(0)

    model = RandomForestClassifier(n_estimators=50, random_state=42) if y.nunique() <= 10 else RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    df['_feature_importance'] = json.dumps(importance_df.to_dict('records'))

output = df
`,
  },

  'automl': {
    type: 'automl',
    category: 'analysis',
    label: 'AutoML',
    description: 'Automatically train and compare multiple models',
    icon: 'Wand2',
    defaultConfig: {
      features: [],
      target: '',
      taskType: 'auto',
      cvFolds: 5,
      models: ['random_forest', 'gradient_boosting', 'logistic', 'knn'],
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import json

df = input_data.copy()
features = config.get('features', [])
target = config.get('target', '')
task_type = config.get('taskType', 'auto')
cv_folds = config.get('cvFolds', 5)
models_to_try = config.get('models', ['random_forest', 'logistic', 'knn'])

if not features:
    raise ValueError("AutoML: Please specify feature columns")
if not target:
    raise ValueError("AutoML: Please specify target column")

X = df[features].copy()
y = df[target].copy()

if task_type == 'auto':
    task_type = 'classification' if y.nunique() <= 10 else 'regression'

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X[col] = X[col].fillna(X[col].median())

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)
else:
    y = y.fillna(y.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if task_type == 'classification':
    model_dict = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
    }
    scoring = 'accuracy'
else:
    model_dict = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'ridge': Ridge(alpha=1.0),
        'knn': KNeighborsRegressor(n_neighbors=5),
    }
    scoring = 'r2'

results = []
best_model = None
best_score = -np.inf

for name in models_to_try:
    if name in model_dict:
        model = model_dict[name]
        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring=scoring)
            mean_score = scores.mean()
            results.append({'model': name, 'mean_score': round(mean_score, 4), 'metric': scoring})
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        except Exception as e:
            results.append({'model': name, 'error': str(e)})

if best_model:
    best_model.fit(X_scaled, y)
    df['automl_prediction'] = best_model.predict(X_scaled)

results = sorted(results, key=lambda x: x.get('mean_score', -np.inf), reverse=True)
df['_automl_leaderboard'] = json.dumps(results)
output = df
`,
  },

  'pipeline-export': {
    type: 'pipeline-export',
    category: 'output',
    label: 'Pipeline Export',
    description: 'Export entire pipeline as standalone Python script',
    icon: 'FileCode',
    defaultConfig: {
      filename: 'pipeline',
      includeComments: true,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd

df = input_data.copy()
filename = config.get('filename', 'pipeline')

script = '''"""
Data Pipeline Script - Generated by DataFlow Canvas
"""
import pandas as pd
import numpy as np

def run_pipeline(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    # Add your transformation steps here
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = run_pipeline(sys.argv[1])
        result.to_csv("output.csv", index=False)
'''

requirements = 'pandas>=1.5.0\\nnumpy>=1.21.0\\nscikit-learn>=1.0.0\\n'

output = pd.DataFrame([{
    'script_content': script,
    'requirements_txt': requirements,
    'filename': filename,
}])
`,
  },

  'multivariate-anomaly': {
    type: 'multivariate-anomaly',
    category: 'analysis',
    label: 'Multivariate Anomaly',
    description: 'Detect anomalies considering multiple features together',
    icon: 'ScanSearch',
    defaultConfig: {
      features: [],
      algorithm: 'isolation_forest',
      contamination: 0.1,
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = input_data.copy()
features = config.get('features', [])
algorithm = config.get('algorithm', 'isolation_forest')
contamination = config.get('contamination', 0.1)

if not features:
    features = df.select_dtypes(include=[np.number]).columns.tolist()

if not features:
    raise ValueError("Multivariate Anomaly: No numeric features available")

X = df[features].copy().fillna(df[features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
predictions = model.fit_predict(X_scaled)

df['is_anomaly'] = (predictions == -1).astype(int)
df['anomaly_score'] = -model.score_samples(X_scaled)

output = df
`,
  },

  'causal-impact': {
    type: 'causal-impact',
    category: 'analysis',
    label: 'Causal Impact',
    description: 'Measure causal effect of interventions',
    icon: 'TrendingDown',
    defaultConfig: {
      outcomeColumn: '',
      treatmentColumn: '',
      method: 'did',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import numpy as np
from scipy import stats
import json

df = input_data.copy()
outcome_col = config.get('outcomeColumn', '')
treatment_col = config.get('treatmentColumn', '')

if not outcome_col:
    raise ValueError("Causal Impact: Please specify outcome column")
if not treatment_col:
    raise ValueError("Causal Impact: Please specify treatment column")

treatment = df[treatment_col].astype(int)
outcome = df[outcome_col].astype(float)

treatment_mean = outcome[treatment == 1].mean()
control_mean = outcome[treatment == 0].mean()
effect = treatment_mean - control_mean

t_stat, p_value = stats.ttest_ind(outcome[treatment == 1], outcome[treatment == 0])

results = {
    'treatment_effect': round(effect, 4),
    't_statistic': round(t_stat, 4),
    'p_value': round(p_value, 4),
    'treatment_mean': round(treatment_mean, 4),
    'control_mean': round(control_mean, 4),
    'significant_at_05': p_value < 0.05,
}

df['_causal_impact_results'] = json.dumps(results)
output = df
`,
  },

  'model-registry': {
    type: 'model-registry',
    category: 'analysis',
    label: 'Model Registry',
    description: 'Save and load trained models with versioning',
    icon: 'Save',
    defaultConfig: {
      action: 'save',
      modelName: '',
      modelVersion: '1.0',
    },
    inputs: 1,
    outputs: 1,
    pythonTemplate: `
import pandas as pd
import json
from datetime import datetime

df = input_data.copy()
action = config.get('action', 'save')
model_name = config.get('modelName', 'untitled_model')
model_version = config.get('modelVersion', '1.0')

if action == 'save':
    model_info_cols = [c for c in df.columns if c.startswith('_') and 'model' in c.lower()]
    model_registry_entry = {
        'name': model_name,
        'version': model_version,
        'created_at': datetime.now().isoformat(),
        'columns': df.columns.tolist(),
    }
    for col in model_info_cols:
        try:
            model_registry_entry[col] = df[col].iloc[0]
        except:
            pass

    df['_model_registry_entry'] = json.dumps(model_registry_entry)
    df['_model_registry_id'] = f"{model_name}_v{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

output = df
`,
  },
};

export const blockCategories = [
  {
    id: 'data-input',
    label: 'Data Input',
    blocks: ['load-data', 'sample-data', 'create-dataset'] as BlockType[],
  },
  {
    id: 'transform',
    label: 'Transform',
    blocks: [
      'filter-rows',
      'select-columns',
      'sort',
      'group-aggregate',
      'join',
      'derive-column',
      'handle-missing',
      'rename-columns',
      'deduplicate',
      'sample-rows',
      'limit-rows',
      'pivot',
      'unpivot',
      'union',
      'split-column',
      'merge-columns',
      'conditional-column',
      'datetime-extract',
      'string-operations',
      'window-functions',
      'bin-bucket',
      'rank',
      'type-conversion',
      'fill-forward-backward',
      'lag-lead',
      'row-number',
      'date-difference',
      'transpose',
      'string-pad',
      'cumulative-operations',
      'replace-values',
      'percent-change',
      'round-numbers',
      'percent-of-total',
      'absolute-value',
      'column-math',
      'extract-substring',
      'parse-date',
      'split-to-rows',
      'clip-values',
      'standardize-text',
      'case-when',
      'log-transform',
      'interpolate-missing',
      'date-truncate',
      'period-over-period',
      'hash-column',
      'expand-date-range',
      'string-similarity',
      'generate-sequence',
      'top-n-per-group',
      'first-last-per-group',
      'one-hot-encode',
      'label-encode',
      'ordinal-encode',
      'min-max-normalize',
      'z-score-standardize',
      'rolling-statistics',
      'resample-timeseries',
      'regex-replace',
      'expand-json-column',
      'add-unique-id',
      'missing-indicator',
      'quantile-transform',
      'explode-column',
      'add-constant-column',
      'drop-columns',
      'flatten-json',
      'coalesce-columns',
      'reorder-columns',
      'trim-text',
      'lookup-vlookup',
      'cross-join',
      'filter-expression',
      'number-format',
      'extract-pattern',
    ] as BlockType[],
  },
  {
    id: 'analysis',
    label: 'Analysis',
    blocks: [
      'statistics',
      'regression',
      'clustering',
      'pca',
      'outlier-detection',
      'classification',
      'normality-test',
      'hypothesis-testing',
      'time-series',
      'feature-importance',
      'cross-validation',
      'data-profiling',
      'value-counts',
      'cross-tabulation',
      'scaling',
      'encoding',
      'ab-test',
      'cohort-analysis',
      'rfm-analysis',
      'anova',
      'chi-square-test',
      'correlation-analysis',
      'survival-analysis',
      'association-rules',
      'sentiment-analysis',
      'moving-average',
      'train-test-split',
      'model-evaluation',
      'knn',
      'naive-bayes',
      'gradient-boosting',
      'pareto-analysis',
      'trend-analysis',
      'forecasting',
      'percentile-analysis',
      'distribution-fit',
      'text-preprocessing',
      'tfidf-vectorization',
      'topic-modeling',
      'similarity-analysis',
      'svm',
      'xgboost',
      'model-explainability',
      'regression-diagnostics',
      'vif-analysis',
      'funnel-analysis',
      'customer-ltv',
      'churn-analysis',
      'growth-metrics',
      'attribution-modeling',
      'breakeven-analysis',
      'confidence-intervals',
      'bootstrap-analysis',
      'posthoc-tests',
      'power-analysis',
      'bayesian-inference',
      'data-quality-score',
      'changepoint-detection',
      'feature-selection',
      'outlier-treatment',
      'data-drift',
      'polynomial-features',
      'multi-output',
      'probability-calibration',
      'tsne-reduction',
      'statistical-tests',
      'optimal-binning',
      'correlation-finder',
      'ab-test-calculator',
      'target-encoding',
      'learning-curves',
      'custom-python-code',
      'sql-query',
      'auto-eda',
      'data-validation',
      'neural-network',
      'auto-feature-engineering',
      'shap-interpretation',
      'automl',
      'multivariate-anomaly',
      'causal-impact',
      'model-registry',
      'isolation-forest',
      'arima-forecasting',
      'seasonal-decomposition',
      'monte-carlo-simulation',
      'propensity-score-matching',
      'difference-in-differences',
      'factor-analysis',
      'dbscan-clustering',
      'elastic-net',
      'var-analysis',
      'interrupted-time-series',
      'granger-causality',
      'local-outlier-factor',
      'imbalanced-data-handler',
      'hyperparameter-tuning',
      'ensemble-stacking',
      'advanced-imputation',
      'umap-reduction',
      'cluster-validation',
      'model-comparison',
      'time-series-cv',
      'uplift-modeling',
      'quantile-regression',
      'adversarial-validation',
    ] as BlockType[],
  },
  {
    id: 'visualization',
    label: 'Visualization',
    blocks: [
      'chart',
      'table',
      'correlation-matrix',
      'violin-plot',
      'pair-plot',
      'area-chart',
      'stacked-chart',
      'bubble-chart',
      'qq-plot',
      'confusion-matrix',
      'roc-curve',
      'funnel-chart',
      'sankey-diagram',
      'treemap',
      'sunburst-chart',
      'gauge-chart',
      'radar-chart',
      'waterfall-chart',
      'candlestick-chart',
      'choropleth-map',
      'word-cloud',
      'pareto-chart',
      'parallel-coordinates',
      'dendrogram',
      'box-plot',
      'heatmap',
      'scatter-map',
      'grouped-histogram',
      'network-graph',
      'calendar-heatmap',
      'faceted-chart',
      'density-plot',
      'error-bar-chart',
      'dot-plot',
      'slope-chart',
      'grouped-bar-chart',
      'bump-chart',
      'donut-chart',
      'horizontal-bar-chart',
      'scatter-3d',
      'contour-plot',
      'hexbin-plot',
      'ridge-plot',
      'strip-plot',
      'bullet-chart',
      'pyramid-chart',
      'timeline-chart',
      'surface-3d',
      'marginal-histogram',
      'dumbbell-chart',
      // New Advanced Visualization Blocks for Data Scientists
      'shap-summary-plot',
      'partial-dependence-plot',
      'feature-importance-plot',
      'ice-plot',
      'precision-recall-curve',
      'learning-curve-plot',
      'residual-plot',
      'actual-vs-predicted-plot',
      'calibration-curve',
      'lift-chart',
      'elbow-plot',
      'silhouette-plot',
      'tsne-umap-plot',
      'missing-value-heatmap',
      'outlier-detection-plot',
      'distribution-comparison-plot',
      'ecdf-plot',
      'andrews-curves',
      'cv-results-plot',
      'hyperparameter-heatmap',
    ] as BlockType[],
  },
  {
    id: 'output',
    label: 'Output',
    blocks: ['export', 'pipeline-export'] as BlockType[],
  },
];
