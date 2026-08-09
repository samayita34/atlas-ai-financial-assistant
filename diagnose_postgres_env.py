from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.database.database import AsyncSessionLocal

# Every statement below is read-only (SELECT / SHOW). Nothing is created,
# dropped, or altered. Session is never committed with any write.
QUERIES: list[tuple[str, str]] = [
    ("PostgreSQL server version", "SELECT version()"),
    ("Current database", "SELECT current_database()"),
    ("Current connected user", "SELECT current_user"),
    ("Server-side listen address (NULL = local Unix socket connection)",
     "SELECT inet_server_addr()"),
    ("Server-side port", "SELECT inet_server_port()"),
    ("Configured port (server setting)", "SHOW port"),
    ("Data directory on the server filesystem", "SHOW data_directory"),
    ("Server config file location", "SHOW config_file"),
]

# Whether the vector extension's control/files exist on the server at all
# (this does NOT install or touch anything -- pg_available_extensions only
# lists what's present on disk vs. what pg_extension has installed).
EXTENSION_AVAILABILITY_QUERY = """
    SELECT name, default_version, installed_version, comment
    FROM pg_available_extensions
    WHERE name = 'vector'
"""

# Per-version detail, including whether pgvector is marked "trusted"
# (a trusted extension can be installed by a non-superuser database owner;
# a non-trusted one requires superuser).
EXTENSION_VERSION_DETAIL_QUERY = """
    SELECT name, version, superuser, trusted, relocatable, schema
    FROM pg_available_extension_versions
    WHERE name = 'vector'
"""

# Current user's own privilege level.
PERMISSION_QUERY = """
    SELECT
        rolname,
        rolsuper AS is_superuser,
        rolcreaterole AS can_create_role,
        rolcreatedb AS can_create_db
    FROM pg_roles
    WHERE rolname = current_user
"""

DATABASE_CREATE_PRIVILEGE_QUERY = """
    SELECT has_database_privilege(current_user, current_database(), 'CREATE') AS can_create_in_db
"""


async def main() -> None:
    print("=== PostgreSQL environment diagnostics (read-only) ===\n")

    async with AsyncSessionLocal() as session:
        for label, query in QUERIES:
            result = (await session.execute(text(query))).scalar()
            print(f"{label}: {result}")

        print()
        print("--- pgvector availability on this server ---")
        avail_rows = (
            await session.execute(text(EXTENSION_AVAILABILITY_QUERY))
        ).fetchall()
        if avail_rows:
            for row in avail_rows:
                print(
                    f"name={row.name} default_version={row.default_version} "
                    f"installed_version={row.installed_version} comment={row.comment}"
                )
        else:
            print(
                "No 'vector' entry found in pg_available_extensions -- the "
                "pgvector extension files are NOT installed on this "
                "PostgreSQL server at all."
            )

        print()
        print("--- pgvector version/trust detail (if listed above) ---")
        detail_rows = (
            await session.execute(text(EXTENSION_VERSION_DETAIL_QUERY))
        ).fetchall()
        if detail_rows:
            for row in detail_rows:
                print(
                    f"version={row.version} requires_superuser={row.superuser} "
                    f"trusted={row.trusted} relocatable={row.relocatable} "
                    f"schema={row.schema}"
                )
        else:
            print("No version detail available (consistent with 'not installed' above).")

        print()
        print("--- current user's install permissions ---")
        perm_row = (await session.execute(text(PERMISSION_QUERY))).fetchone()
        if perm_row:
            print(
                f"role={perm_row.rolname} is_superuser={perm_row.is_superuser} "
                f"can_create_role={perm_row.can_create_role} "
                f"can_create_db={perm_row.can_create_db}"
            )
        create_priv = (
            await session.execute(text(DATABASE_CREATE_PRIVILEGE_QUERY))
        ).scalar()
        print(f"can_create_in_current_database={create_priv}")

    print()
    print("=== End diagnostics -- no changes were made ===")


if __name__ == "__main__":
    asyncio.run(main())