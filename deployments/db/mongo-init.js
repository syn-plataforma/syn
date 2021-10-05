// Create REST API database and user.
use syn_rest_api;

// Create user.
db.createUser(
        {
            user: "**********",
            pwd: "**********",
            roles: [
                {
                    role: "readWrite",
                    db: "syn_rest_api"
                }
            ]
        }
);

// Create REST API collections.
db.createCollection('training_parameters');
db.createCollection('users');

use admin;

db.createUser(
    {
        user: "**********",
        pwd: "**********",
        roles: ["readAnyDatabase"]
    }
)