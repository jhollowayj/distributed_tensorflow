-- Table: public.tf_united_episode_stats

-- DROP TABLE public.tf_united_episode_stats;

CREATE TABLE public.tf_united_episode_stats
(
  id serial NOT NULL,
  learner_uuid uuid NOT NULL, -- universally unique ID.
  episode integer NOT NULL, -- specific run through the game (should be unique on the uuid above
  steps_in_episode integer NOT NULL, -- number of steps that game took to finish (or die)
  total_reward real NOT NULL, -- Reward gained that game
  q_max real NOT NULL, -- Max q values seen across all actions, all steps
  q_min real NOT NULL, -- Min q values seen across all actions, all steps
  avg_action_value_n real NOT NULL, -- Avg value of the North commands across that game (checking for learning one action)
  avg_action_value_e real NOT NULL, -- Avg value of the South commands across that game (checking for learning one action)
  avg_action_value_s real NOT NULL, -- Avg value of the East commands across that game (checking for learning one action)
  avg_action_value_w real NOT NULL, -- Avg value of the West commands across that game (checking for learning one action)
  mean_cost real NOT NULL, -- Average cost on the network (magic!)
  end_epsilon real NOT NULL, -- Epsilon value for that game (in epsilon greedy)
  did_win boolean NOT NULL, -- Boolean flag of did win or not.
  date_run timestamp without time zone DEFAULT timezone('utc'::text, now()),
  -- CONSTRAINT tf_united_episode_stats_pkey PRIMARY KEY (id) -- Delete this line when restarting a new table
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.tf_united_episode_stats
  OWNER TO postgres;
COMMENT ON COLUMN public.tf_united_episode_stats.learner_uuid IS 'universally unique ID.';
COMMENT ON COLUMN public.tf_united_episode_stats.episode IS 'specific run through the game (should be unique on the uuid above';
COMMENT ON COLUMN public.tf_united_episode_stats.steps_in_episode IS 'number of steps that game took to finish (or die)';
COMMENT ON COLUMN public.tf_united_episode_stats.total_reward IS 'Reward gained that game';
COMMENT ON COLUMN public.tf_united_episode_stats.q_max IS 'Max q values seen across all actions, all steps';
COMMENT ON COLUMN public.tf_united_episode_stats.q_min IS 'Min q values seen across all actions, all steps';
COMMENT ON COLUMN public.tf_united_episode_stats.avg_action_value_n IS 'Avg value of the North commands across that game (checking for learning one action)';
COMMENT ON COLUMN public.tf_united_episode_stats.avg_action_value_e IS 'Avg value of the South commands across that game (checking for learning one action)';
COMMENT ON COLUMN public.tf_united_episode_stats.avg_action_value_s IS 'Avg value of the East commands across that game (checking for learning one action)';
COMMENT ON COLUMN public.tf_united_episode_stats.avg_action_value_w IS 'Avg value of the West commands across that game (checking for learning one action)';
COMMENT ON COLUMN public.tf_united_episode_stats.mean_cost IS 'Average cost on the network (magic!)';
COMMENT ON COLUMN public.tf_united_episode_stats.end_epsilon IS 'Epsilon value for that game (in epsilon greedy)';
COMMENT ON COLUMN public.tf_united_episode_stats.did_win IS 'Boolean flag of did win or not.';

