-- Widen the preserved GDACS country text without changing existing values.
-- Safe to rerun: TEXT is accepted as the already-migrated state.

BEGIN;

DO $migration$
DECLARE
    current_data_type text;
    current_maximum_length integer;
BEGIN
    SELECT data_type, character_maximum_length
      INTO current_data_type, current_maximum_length
      FROM information_schema.columns
     WHERE table_schema = 'public'
       AND table_name = 'events'
       AND column_name = 'country';

    IF NOT FOUND THEN
        RAISE NOTICE 'public.events.country does not exist; no migration required';
    ELSIF current_data_type = 'text' THEN
        NULL;
    ELSIF current_data_type = 'character varying'
          AND current_maximum_length = 128 THEN
        ALTER TABLE public.events
            ALTER COLUMN country TYPE text;
    ELSE
        RAISE EXCEPTION
            'Unexpected public.events.country type: %, maximum length: %',
            current_data_type,
            current_maximum_length;
    END IF;
END;
$migration$;

COMMIT;
