/**
 * Test Studio page — routes /studio (standalone) and /dataset/studio/:id
 * (legacy, pre-filled with a dataset).
 *
 * Reads the dataset id from the URL (`:id` param) OR the `?dataset=` query
 * param and passes it as `preselectDataset` to StudioShell: a blank page if
 * neither is set, otherwise that LoRA is pre-checked in the picker.
 *
 * Gated on `caps.studio_visible` (ComfyUI reachable): the nav link already
 * hides the entry, but this guards direct URL access too.
 */
import { Link, useParams, useSearchParams } from 'react-router';
import EmptyState from '../components/common/EmptyState';
import { StudioIcon } from '../components/common/icons';
import { PRIMARY_BUTTON } from '../components/common/surfaces';
import { useCapabilities } from '../context/CapabilitiesContext';
import StudioShell from '../components/dataset/studio/StudioShell';

export default function StudioPage() {
  const { id } = useParams();
  const [sp] = useSearchParams();
  const { caps } = useCapabilities();
  // /dataset/studio/:id (legacy), or /studio?dataset=… (launcher), or nothing (standalone).
  const preselectDataset = id || sp.get('dataset') || null;
  const preselectFamily = sp.get('family') || null;
  // `?base=` — the base model to open on, as ComfyUI's loader names it. Sent by
  // the full-model card in Checkpoints & LoRAs: arriving there means "test THIS
  // model", and a full model that is not preselected is a model you have to go
  // and find in a dropdown among every checkpoint on the machine. It also
  // re-seeds CFG/steps, which is the point for an undistilled base — the
  // family's few-step defaults render mush on one.
  const preselectBase = sp.get('base') || null;

  // Not a page header: this is the page having nothing to show, which is what
  // EmptyState is for. It also stops the fallback claiming an <h1> that the
  // real page never renders — two different documents answering to one route.
  if (!caps.studio_visible) {
    return (
      <EmptyState
        icon={<StudioIcon className="h-5 w-5" />}
        title="Test Studio needs ComfyUI"
        action={(
          <Link to="/settings/local-tools?focus=comfyui-api-url" className={PRIMARY_BUTTON}>
            Open Settings
          </Link>
        )}
      >
        The Studio renders every test on your own ComfyUI. Point the app at it in
        Settings and this page fills in.
      </EmptyState>
    );
  }

  // pb-24: StudioActionBar is a fixed bottom bar (Run button + section shortcuts) —
  // leaves room so it never covers the last row of results.
  return (
    <div className="pb-24">
      <StudioShell preselectDataset={preselectDataset} preselectFamily={preselectFamily}
        preselectBase={preselectBase} />
    </div>
  );
}
