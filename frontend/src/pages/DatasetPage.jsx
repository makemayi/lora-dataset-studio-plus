/**
 * Dataset Maker page — build a face dataset for LoRA character training:
 * generate Klein variations from a reference, import real photos, curate,
 * caption (Qwen3-VL), and export a training-ready ZIP.
 */
import { useDataset } from '../hooks/useDataset';
import DatasetListPanel from '../components/dataset/DatasetListPanel';
import DatasetWorkspace from '../components/dataset/DatasetWorkspace';
import VideoDatasetsPanel from '../components/videobank/VideoDatasetsPanel';

export default function DatasetPage() {
  const ds = useDataset();
  return (
    /* No measure of its own since 2026-08-10. This page carried `max-w-6xl`
       (1152px) INSIDE the shell, so making the shell wide changed nothing here
       — the library still drew four columns in the middle of a 1920px screen
       and the workspace still put its rail and its grid in a 1150px box. The
       shell (App.jsx) owns the measure for every route; a page that wants a
       narrower one caps its own CONTENT, the way the empty-state hero and the
       creation form below do. */
    <div>
      {ds.currentId ? (
        <DatasetWorkspace ds={ds} onBack={() => ds.setCurrentId(null)} />
      ) : (
        <div className="flex flex-col gap-4">
          <DatasetListPanel datasets={ds.datasets} onOpen={ds.open} onCreate={ds.create}
            onDelete={ds.deleteDataset} onRestore={ds.importBackup}
            onExportZip={ds.exportZipFor} onExportBackup={ds.exportBackupFor}
            onSettingsSave={ds.updateSettingsFor}
            backup={{
              start: ds.backupEverything, job: ds.backupJob,
              download: ds.downloadBackup, openFolder: ds.openBackupsFolder,
              dismiss: ds.dismissBackup,
              restoreJob: ds.restoreJob, dismissRestore: ds.dismissRestore,
            }} />
          {/* Video training sets live in the SAME library, below the image ones —
              they are datasets, and a second page for them would be a second
              place to remember. The panel renders nothing at all until one
              exists, so someone who never touched the video lane never pays a
              permanently empty section. */}
          <VideoDatasetsPanel />
        </div>
      )}
    </div>
  );
}
