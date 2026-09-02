import { useEffect, useState } from 'react'
import {
  RiCheckDoubleLine,
  RiCheckLine,
  RiClipboardLine,
  RiDownload2Line,
  RiFileCodeLine,
  RiFolderDownloadLine,
  RiHistoryLine,
  RiRefreshLine,
  RiShieldCheckLine,
  RiSparklingLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Spinner } from '@/components/ui/spinner'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { EpisodeSummary, ExportHistoryItem, ExportOutItem } from '@/types'

const EXPORT_PROFILES = [
  {
    id: 'training',
    name: 'Training Split',
    tag: 'Primary',
    tagColor: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
    description:
      'Train and validation splits with accepted & edited ground-truth labels for ASR fine-tuning.',
    splits: ['train', 'val'],
    dispositions: ['accepted_unchanged', 'edited'],
  },
  {
    id: 'gold',
    name: 'Gold Benchmark',
    tag: 'Evaluation',
    tagColor: 'bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/20',
    description:
      'Frozen test split retaining seed systems for reproducible academic and production benchmarking.',
    splits: ['test'],
    dispositions: ['accepted_unchanged', 'edited'],
  },
  {
    id: 'analytics',
    name: 'Full Analytics',
    tag: 'Diagnostic',
    tagColor: 'bg-purple-500/15 text-purple-600 dark:text-purple-400 border-purple-500/20',
    description:
      'Complete corpus with all hypotheses, word-level timestamps, logprobs, and agreement scores.',
    splits: ['train', 'val', 'test', 'unassigned'],
    dispositions: ['All dispositions'],
  },
  {
    id: 'error_mining',
    name: 'Error Mining',
    tag: 'Quality QA',
    tagColor: 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/20',
    description:
      'Segments flagged as uncertain or unusable audio to isolate edge cases and train filters.',
    splits: ['All splits'],
    dispositions: ['unusable_audio', 'uncertain'],
  },
] as const

function formatBytes(bytes: number) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}

export function ExportView() {
  const [selectedKind, setSelectedKind] = useState<string>('training')
  const [selectedEpisode, setSelectedEpisode] = useState<string>('')
  const [labelVersion, setLabelVersion] = useState<string>('v1')

  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [isExporting, setIsExporting] = useState(false)
  const [exportResults, setExportResults] = useState<ExportOutItem[]>([])
  const [exportHistory, setExportHistory] = useState<ExportHistoryItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(true)

  const [copiedHash, setCopiedHash] = useState<string | null>(null)

  const loadData = async () => {
    setLoadingHistory(true)
    try {
      const [eps, hist] = await Promise.all([api.listEpisodes(), api.getExportHistory()])
      setEpisodes(eps)
      setExportHistory(hist)
    } catch {
      // ignore
    } finally {
      setLoadingHistory(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const handleRunExport = async () => {
    setIsExporting(true)
    try {
      const resp = await api.runExport({
        kind: selectedKind,
        episode: selectedEpisode.trim() || undefined,
        label_version: labelVersion.trim() || undefined,
      })

      setExportResults(resp.results)
      toast.success(
        `Successfully exported ${resp.results.length} dataset${resp.results.length > 1 ? 's' : ''}`,
        {
          description: `${resp.results.reduce((acc, r) => acc + r.row_count, 0)} total rows generated.`,
        },
      )
      // Refresh history to show newly created files
      api.getExportHistory().then(setExportHistory)
    } catch (err: any) {
      toast.error(err.message || 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedHash(id)
    toast.info('Copied to clipboard')
    setTimeout(() => setCopiedHash(null), 2500)
  }

  return (
    <div className="scrollbar-thin flex-1 overflow-y-auto bg-background p-6 space-y-6">
      {/* View Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b pb-4">
        <div>
          <div className="flex items-center gap-2">
            <RiFolderDownloadLine className="size-6 text-primary" />
            <h1 className="font-heading text-xl font-bold tracking-tight">Dataset Export</h1>
            <Badge variant="secondary" className="font-mono text-xs">
              JSONL &amp; Manifest
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Export reproducible, hash-verified datasets directly to disk and download in-browser.
          </p>
        </div>

        <Button variant="outline" size="sm" onClick={loadData} className="gap-1.5 h-8">
          <RiRefreshLine className="size-3.5" />
          Refresh
        </Button>
      </div>

      {/* STEP 1: Select Export Profile */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-heading text-xs font-semibold tracking-wider text-muted-foreground uppercase">
            1. Select Dataset Kind
          </span>
          <Button
            variant={selectedKind === 'all' ? 'default' : 'outline'}
            size="sm"
            className="h-7 text-xs"
            onClick={() => setSelectedKind('all')}
          >
            Export All Profiles Simultaneously
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {EXPORT_PROFILES.map((profile) => {
            const isSelected = selectedKind === profile.id

            return (
              <Card
                key={profile.id}
                onClick={() => setSelectedKind(profile.id)}
                className={cn(
                  'cursor-pointer transition-all border',
                  isSelected
                    ? 'border-primary ring-2 ring-primary/20 bg-primary/5'
                    : 'border-border/60 hover:border-primary/40 hover:bg-muted/30',
                )}
              >
                <CardHeader className="p-4 pb-2 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle className="text-sm font-bold font-heading">{profile.name}</CardTitle>
                    <span
                      className={cn(
                        'rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase border',
                        profile.tagColor,
                      )}
                    >
                      {profile.tag}
                    </span>
                  </div>
                  <CardDescription className="text-xs line-clamp-2">
                    {profile.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  <div className="mt-2 space-y-1 text-[11px] font-mono text-muted-foreground">
                    <div>
                      Splits: <span className="text-foreground">{profile.splits.join(', ')}</span>
                    </div>
                    <div>
                      Dispositions:{' '}
                      <span className="text-foreground">{profile.dispositions.join(', ')}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* STEP 2: Filters & Options */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">2. Export Filters &amp; Configuration</CardTitle>
          <CardDescription className="text-xs">
            Optionally narrow the export to a single episode or specify a label version.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 items-end">
            <div className="space-y-1.5">
              <Label className="text-xs font-semibold">Filter Episode (Optional)</Label>
              <select
                value={selectedEpisode}
                onChange={(e) => setSelectedEpisode(e.target.value)}
                className="w-full h-9 rounded-md border border-input bg-background px-3 py-1 text-xs shadow-xs focus:outline-hidden focus:ring-1 focus:ring-ring"
              >
                <option value="">All Episodes in Corpus</option>
                {episodes.map((ep) => (
                  <option key={ep.id} value={ep.external_id}>
                    {ep.title ? `${ep.external_id} - ${ep.title}` : ep.external_id} ({ep.segment_count}{' '}
                    segs)
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs font-semibold">Label Version</Label>
              <Input
                value={labelVersion}
                onChange={(e) => setLabelVersion(e.target.value)}
                placeholder="v1"
                className="h-9 text-xs font-mono"
              />
            </div>

            <div>
              <Button
                size="default"
                disabled={isExporting}
                onClick={handleRunExport}
                className="w-full gap-2"
              >
                {isExporting ? <Spinner /> : <RiSparklingLine className="size-4" />}
                {isExporting
                  ? 'Generating Export…'
                  : `Export ${selectedKind === 'all' ? 'All Datasets' : selectedKind.toUpperCase()}`}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* STEP 3: Live Export Results */}
      {exportResults.length > 0 && (
        <Card className="border-primary/40 bg-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <RiCheckDoubleLine className="size-5 text-emerald-500" />
                <CardTitle className="text-sm font-semibold">Export Generated Successfully</CardTitle>
              </div>
              <Badge variant="outline" className="font-mono text-xs text-emerald-600 dark:text-emerald-400">
                {exportResults.reduce((sum, r) => sum + r.row_count, 0)} Total Rows
              </Badge>
            </div>
            <CardDescription className="text-xs">
              Files have been written to disk and are ready for immediate download.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {exportResults.map((result) => {
              const splits = Object.entries(result.row_counts_by_split)
              const sha256 = result.manifest?.files?.[0]?.sha256 || ''

              return (
                <div key={result.kind} className="rounded-lg border p-4 space-y-3 bg-background">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-heading text-sm font-bold uppercase">{result.kind}</span>
                      <Badge variant="secondary" className="font-mono text-xs">
                        {result.row_count} rows
                      </Badge>
                      {splits.map(([sName, sCount]) => (
                        <span
                          key={sName}
                          className="rounded bg-muted px-2 py-0.5 font-mono text-[10px] text-muted-foreground"
                        >
                          {sName}: {sCount}
                        </span>
                      ))}
                    </div>

                    <div className="flex items-center gap-2">
                      <a
                        href={api.getExportDownloadUrl(result.kind, result.data_filename)}
                        download={result.data_filename}
                      >
                        <Button variant="default" size="sm" className="h-8 gap-1.5 text-xs">
                          <RiDownload2Line className="size-3.5" />
                          Download JSONL
                        </Button>
                      </a>
                      <a
                        href={api.getExportDownloadUrl(result.kind, result.manifest_filename)}
                        download={result.manifest_filename}
                      >
                        <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs">
                          <RiFileCodeLine className="size-3.5" />
                          Manifest
                        </Button>
                      </a>
                    </div>
                  </div>

                  {/* SHA-256 Checksum */}
                  {sha256 && (
                    <div className="flex items-center justify-between rounded bg-muted/40 px-3 py-2 text-xs font-mono">
                      <div className="flex items-center gap-2 truncate text-muted-foreground">
                        <RiShieldCheckLine className="size-4 text-emerald-500 shrink-0" />
                        <span className="text-[11px] truncate">SHA-256: {sha256}</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => handleCopy(sha256, result.kind)}
                        title="Copy SHA-256"
                      >
                        {copiedHash === result.kind ? (
                          <RiCheckLine className="size-3.5 text-emerald-500" />
                        ) : (
                          <RiClipboardLine className="size-3.5" />
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              )
            })}
          </CardContent>
        </Card>
      )}

      {/* STEP 4: Recent Exports on Disk */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <RiHistoryLine className="size-4 text-primary" />
            <CardTitle className="text-sm font-semibold">Available Exports on Disk</CardTitle>
          </div>
          <CardDescription className="text-xs">
            Previous datasets generated beneath the configured export directory.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadingHistory ? (
            <div className="flex items-center justify-center py-6 gap-2 text-xs text-muted-foreground">
              <Spinner /> Loading past export history…
            </div>
          ) : exportHistory.length === 0 ? (
            <div className="py-8 text-center text-xs text-muted-foreground">
              No previous exports found on disk. Click &quot;Export Dataset&quot; above to generate one.
            </div>
          ) : (
            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs font-semibold">Kind</TableHead>
                    <TableHead className="text-xs font-semibold">File</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Rows</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Size</TableHead>
                    <TableHead className="text-xs font-semibold">Exported At</TableHead>
                    <TableHead className="text-xs font-semibold text-right">Download</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {exportHistory.map((item) => (
                    <TableRow key={item.kind}>
                      <TableCell className="font-heading text-xs uppercase font-bold">
                        {item.kind}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{item.data_filename}</TableCell>
                      <TableCell className="text-right font-mono text-xs font-medium">
                        {item.row_count.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs">
                        {formatBytes(item.file_bytes)}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {item.exported_at ? new Date(item.exported_at).toLocaleString() : '--'}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <a
                            href={api.getExportDownloadUrl(item.kind, item.data_filename)}
                            download={item.data_filename}
                          >
                            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs gap-1">
                              <RiDownload2Line className="size-3" /> JSONL
                            </Button>
                          </a>
                          <a
                            href={api.getExportDownloadUrl(item.kind, item.manifest_filename)}
                            download={item.manifest_filename}
                          >
                            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs gap-1">
                              <RiFileCodeLine className="size-3" /> Manifest
                            </Button>
                          </a>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
