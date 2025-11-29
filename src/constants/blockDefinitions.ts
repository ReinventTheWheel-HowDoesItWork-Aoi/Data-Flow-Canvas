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
    ] as BlockType[],
  },
  {
    id: 'output',
    label: 'Output',
    blocks: ['export'] as BlockType[],
  },
];
