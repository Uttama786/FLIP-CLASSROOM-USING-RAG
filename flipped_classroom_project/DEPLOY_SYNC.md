# Syncing Videos & Notes: Localhost → Render

Render does **not** copy your laptop automatically. You have two separate stores:

| What | Localhost | Render |
|------|-----------|--------|
| Database | `db.sqlite3` | PostgreSQL (empty or partial) |
| Video/note files | `media/` folder on disk | Ephemeral disk (lost on redeploy) unless **Cloudinary** |

That is why you see many items locally but fewer on Render.

## One-time setup on Render

1. Create a free [Cloudinary](https://cloudinary.com) account.
2. In **Render → fliplearn → Environment**, add:
   - `CLOUDINARY_URL` = `cloudinary://API_KEY:API_SECRET@CLOUD_NAME`
3. Redeploy after saving.

## Sync your content (recommended)

Run these on your **PC** inside `flipped_classroom_project/`:

### Step 1 — Export database rows (videos, notes, quizzes)

```powershell
python manage.py export_content_fixture
```

This creates `fixtures/fliplearn_content.json`.

Commit that file (or upload it to Render Shell).

### Step 2 — Upload all local files to Cloudinary

Add to `.env` (same Cloudinary URL as Render):

```
CLOUDINARY_URL=cloudinary://...
```

Then:

```powershell
python manage.py push_media_to_cloudinary
```

Optional dry-run:

```powershell
python manage.py push_media_to_cloudinary --dry-run
```

### Step 3 — Load content on Render

**Automatic (recommended):** Every deploy runs `load_content_fixture` in `startup.py` if the DB has fewer than 50 videos.

**Manual** (Render Shell):

```bash
python manage.py load_content_fixture
# or if partial/broken data:
python manage.py load_content_fixture --force
```

Requires uploader users `admin`, `teacher`, `prof_sharma` — the command creates them before `loaddata`.

If you uploaded media in Step 2 against the **local** DB, re-run Step 2 with Render's external `DATABASE_URL` in the shell environment so Cloudinary URLs are stored in production Postgres.

### Alternative: point upload at Render DB from your PC

```powershell
$env:DATABASE_URL="postgresql://..."   # External Database URL from Render
$env:CLOUDINARY_URL="cloudinary://..."
python manage.py push_media_to_cloudinary
python manage.py export_content_fixture
# then loaddata on Render if needed
```

## After deploy

- New uploads on Render go to Cloudinary automatically (when `CLOUDINARY_URL` is set).
- Do **not** rely on files in `media/` on the server; they disappear on redeploy.
- `media/videos/` is in `.gitignore`, so Git never ships your video files to Render.

## Quick check

Render Shell:

```bash
python manage.py shell -c "from flipped_app.models import VideoLecture, StudyMaterial; print('videos', VideoLecture.objects.count()); print('materials', StudyMaterial.objects.count())"
```

Compare counts with localhost admin dashboard.
