import uuid
import random
import string
from datetime import datetime, timedelta

# Configuration
NUMBER_OF_INSERTS = 3  # Number of INSERT queries to generate
TIMESTAMP_MODE = 'incremental'  # Options: 'fixed', 'random', 'incremental'
TIMESTAMP_RANGE = (-30, 30)  # Random offset range in seconds (used if mode='random')
TIMESTAMP_INCREMENT = 2  # Seconds to increment per record (used if mode='incremental')
START_TIME_OFFSET_HOURS = -9 # Hours to offset from current time (e.g., -4 for 4 hours ago)
transaction_id_prefix = 'exit_'

def generate_unique_id(length, case='lower'):
    """Generate a random string of fixed length for IDs, including prefix, with specified case."""
    # Validate case parameter
    if case not in ['upper', 'lower']:
        raise ValueError("case must be 'upper' or 'lower'")
    
    # Choose character set based on case
    char_set = string.ascii_uppercase + string.digits if case == 'upper' else string.ascii_lowercase + string.digits
    case_func = str.upper if case == 'upper' else str.lower
    
    # Start with UUID, remove hyphens
    uuid_str = case_func(str(uuid.uuid4()).replace('-', ''))
    # Calculate required length after adding prefix
    required_length = length - len(transaction_id_prefix)
    # Use UUID as base
    result = uuid_str[:required_length]
    # If too short, append random characters from char_set
    if len(result) < required_length:
        remaining_length = required_length - len(result)
        random_chars = ''.join(random.choices(char_set, k=remaining_length))
        result += random_chars
    return transaction_id_prefix + result

def generate_timestamp(base_time, mode='random', offset_range=(-10, 10), increment=0, index=0):
    """Generate a timestamp based on the specified mode."""
    if mode == 'fixed':
        return base_time.strftime('%Y-%m-%d %H:%M:%S')
    elif mode == 'incremental':
        return (base_time + timedelta(seconds=increment * index)).strftime('%Y-%m-%d %H:%M:%S')
    else:  # Default: random
        offset = random.randint(offset_range[0], offset_range[1])
        return (base_time + timedelta(seconds=offset)).strftime('%Y-%m-%d %H:%M:%S')

def generate_synthetic_transaction(base_time, counter, timestamp_mode='random', timestamp_range=(-10, 10), timestamp_increment=1, id_case='upper', hashed_pan_case='lower', response_id_z1_case='lower'):
    """Generate a single synthetic transaction record with NOT NULL columns, hashed_pan, and response_id_z1."""
    gateway_mid = '10067061'
    created_by = 'brain'
    updated_by = 'elisheva'

    # Generate dynamic values
    transaction_id = generate_unique_id(length=32, case='upper')  
    hashed_pan = 'd90e9b5733d60c7c3bf4bae50f5005b0a849fd5303b06e1e62bbe92e81c0b39c77da37840ad30db2c7fe434b24f5bc5edfd73a7e9a040e583bae065d7c4accd4' 
    response_id_z1 = generate_unique_id(length=32, case='lower')   
    request_id_a1 = f'S4760378007GSA{counter:015d}'
    request_timestamp = generate_timestamp(base_time, mode=timestamp_mode, offset_range=timestamp_range, increment=timestamp_increment, index=counter-1000)

    # # Validate lengths
    # assert len(transaction_id) == 32, f"transaction_id length is {len(transaction_id)}, expected 32"
    # assert len(hashed_pan) == 128, f"hashed_pan length is {len(hashed_pan)}, expected 128"
    # assert len(response_id_z1) == 32, f"response_id_z1 length is {len(response_id_z1)}, expected 32"

    # Construct the INSERT query with foor mandtories fields only
    query = f"""INSERT INTO brain.transactions
(id, gateway_mid, request_id_a1, request_timestamp, 
response_timestamp, created, created_by, updated, updated_by, hashed_pan, response_id_z1)
VALUES (
    '{transaction_id}', '{gateway_mid}', '{request_id_a1}', '{request_timestamp}', '{request_timestamp}', '{request_timestamp}', '{created_by}', '{request_timestamp}', '{updated_by}', '{hashed_pan}', '{response_id_z1}'
);
"""
    return query

def generate_multiple_inserts(number_of_inserts, timestamp_mode='random', timestamp_range=(-10, 10), timestamp_increment=1, id_case='upper', hashed_pan_case='lower', response_id_z1_case='lower'):
    """Generate multiple synthetic INSERT queries."""
    base_time = datetime.now() + timedelta(hours=START_TIME_OFFSET_HOURS)  # Apply user-defined offset
    queries = []
    for i in range(number_of_inserts):
        query = generate_synthetic_transaction(base_time, 1000 + i, timestamp_mode, timestamp_range, timestamp_increment, id_case, hashed_pan_case, response_id_z1_case)
        queries.append(query)
    return queries

# Generate and print synthetic INSERT queries
synthetic_queries = generate_multiple_inserts(
    number_of_inserts=NUMBER_OF_INSERTS,
    timestamp_mode=TIMESTAMP_MODE,
    timestamp_range=TIMESTAMP_RANGE,
    timestamp_increment=TIMESTAMP_INCREMENT,
    id_case='upper',
    hashed_pan_case='lower',
    response_id_z1_case='lower'
)

# Print queries
for i, query in enumerate(synthetic_queries, 1):
  #  print(f"-- Synthetic Transaction {i}")
    print(query)

# Optionally save to file
# with open('synthetic_transactions.sql', 'w') as f:
#     for query in synthetic_queries:
#         f.write(query + '\n')