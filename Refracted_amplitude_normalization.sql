-- Copyright 2025 Anvar Mohamed Aslam Sha
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Step 1: Create normalized table
CREATE TABLE IF NOT EXISTS amplitude_response_normalized (
    id INT AUTO_INCREMENT PRIMARY KEY,
    simulation_id INT,
    point_index INT,
    amplitude FLOAT,
    amplitude_norm FLOAT,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

-- Step 2: Clean + Normalize amplitudes (Z-score filtering + Min-Max normalization)
INSERT INTO amplitude_response_normalized (simulation_id, point_index, amplitude, amplitude_norm)
SELECT
    ar.simulation_id,
    ar.point_index,
    ar.amplitude,
    (ar.amplitude - mm.min_amp) / NULLIF((mm.max_amp - mm.min_amp), 0) AS amplitude_norm
FROM amplitude_response ar

-- Join for mean and stddev per simulation (for Z-score)
JOIN (
    SELECT
        simulation_id,
        AVG(amplitude) AS mean_amp,
        STDDEV_POP(amplitude) AS std_amp
    FROM amplitude_response
    WHERE amplitude IS NOT NULL
    GROUP BY simulation_id
) stats ON ar.simulation_id = stats.simulation_id

-- Join for Min-Max normalization per simulation
JOIN (
    SELECT
        simulation_id,
        MIN(amplitude) AS min_amp,
        MAX(amplitude) AS max_amp
    FROM amplitude_response
    WHERE amplitude IS NOT NULL
    GROUP BY simulation_id
) mm ON ar.simulation_id = mm.simulation_id

-- Apply Z-score based outlier filter
WHERE ar.amplitude IS NOT NULL
  AND ABS((ar.amplitude - stats.mean_amp) / NULLIF(stats.std_amp, 0)) <= 3;
