-- Table: public.tf_united_game_settings

-- DROP TABLE public.tf_united_game_settings;

CREATE TABLE public.tf_united_game_settings
(
  id serial PRIMARY KEY NOT NULL, -- Primary key
  learner_uuid uuid NOT NULL, -- universally unique ID for the learner (speareate from the DQN value)
  parallel_learning_session_uuid uuid NOT NULL, -- DQN should generate this everytime it is created
  world_id integer NOT NULL, -- id of world (map layout)
  task_id integer NOT NULL, -- id of task (start and end locations)
  agent_id integer NOT NULL, -- id of agent (nsew vs sewn vs ewns, etc)
  max_episode_count integer NOT NULL, -- max number of steps the game could run before being stopped
  annealing_size integer NOT NULL, -- How many trainings we should go before we reaching final_epsilon.  (Linear transition)
  final_epsilon real NOT NULL, -- Final epsilon value to use once annealing_size has been reached
  num_parallel_learners integer NOT NULL, -- shows how many learners were running for that session
  using_experience_replay boolean NOT NULL, -- Boolean for using experience replay vs just training on one game.
  date_run timestamp without time zone DEFAULT timezone('utc'::text, now()),
  CONSTRAINT tf_united_game_settings_pkey PRIMARY KEY (id) -- Delete this line when restarting a new table
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.tf_united_game_settings
  OWNER TO postgres;
COMMENT ON COLUMN public.tf_united_game_settings.id IS 'Primary key';
COMMENT ON COLUMN public.tf_united_game_settings.learner_uuid IS 'universally unique ID for the learner (speareate from the DQN value)';
COMMENT ON COLUMN public.tf_united_game_settings.parallel_learning_session_uuid IS 'DQN should generate this everytime it is created';
COMMENT ON COLUMN public.tf_united_game_settings.world_id IS 'id of world (map layout)';
COMMENT ON COLUMN public.tf_united_game_settings.task_id IS 'id of task (start and end locations)';
COMMENT ON COLUMN public.tf_united_game_settings.agent_id IS 'id of agent (nsew vs sewn vs ewns, etc)';
COMMENT ON COLUMN public.tf_united_game_settings.max_episode_count IS 'max number of steps the game could run before being stopped';
COMMENT ON COLUMN public.tf_united_game_settings.annealing_size IS 'How many trainings we should go before we reaching final_epsilon.  (Linear transition)';
COMMENT ON COLUMN public.tf_united_game_settings.final_epsilon IS 'Final epsilon value to use once annealing_size has been reached';
COMMENT ON COLUMN public.tf_united_game_settings.num_parallel_learners IS 'shows how many learners were running for that session';
COMMENT ON COLUMN public.tf_united_game_settings.using_experience_replay IS 'Boolean for using experience replay vs just training on one game.';

