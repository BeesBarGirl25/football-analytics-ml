    CREATE TABLE IF NOT EXISTS model_version (
      model_version TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      notes TEXT,
      k_final INT,
      pca_components INT,
      feats JSONB
    );
    
    CREATE TABLE IF NOT EXISTS player (
      model_version TEXT NOT NULL,
      player_key TEXT NOT NULL,
      dataset TEXT NOT NULL,
      player TEXT NOT NULL,
      player_position TEXT,
      role_id INT,
      role_name TEXT,
      PRIMARY KEY (model_version, player_key, dataset)
    );
    
    CREATE INDEX IF NOT EXISTS idx_player_name
      ON player (player);
    
    CREATE TABLE IF NOT EXISTS player_embedding (
      model_version TEXT NOT NULL,
      player_key TEXT NOT NULL,
      dataset TEXT NOT NULL,
      embedding DOUBLE PRECISION[] NOT NULL,
      PRIMARY KEY (model_version, player_key, dataset)
    );
    
    CREATE TABLE IF NOT EXISTS player_features (
      model_version TEXT NOT NULL,
      player_key TEXT NOT NULL,
      dataset TEXT NOT NULL,
      features DOUBLE PRECISION[] NOT NULL,
      PRIMARY KEY (model_version, player_key, dataset)
    );
    
    CREATE TABLE IF NOT EXISTS player_neighbour (
      model_version TEXT NOT NULL,
      src_player_key TEXT NOT NULL,
      src_dataset TEXT NOT NULL,
      dst_player_key TEXT NOT NULL,
      dst_dataset TEXT NOT NULL,
      rank INT NOT NULL,
      similarity DOUBLE PRECISION NOT NULL,
      PRIMARY KEY (model_version, src_player_key, src_dataset, rank)
    );
    
    CREATE INDEX IF NOT EXISTS idx_neigh_src
      ON player_neighbour (model_version, src_player_key, src_dataset);
