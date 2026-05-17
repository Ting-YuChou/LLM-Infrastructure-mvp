{{/*
Common chart helpers.
*/}}
{{- define "vllm-inference.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vllm-inference.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vllm-inference.namespace" -}}
{{- default .Release.Namespace .Values.namespace.name -}}
{{- end -}}

{{- define "vllm-inference.labels" -}}
helm.sh/chart: {{ include "vllm-inference.chart" . }}
app.kubernetes.io/name: {{ include "vllm-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app: {{ .Values.selectorLabels.app }}
version: {{ .Values.selectorLabels.version }}
{{- end -}}

{{- define "vllm-inference.selectorLabels" -}}
app: {{ .Values.selectorLabels.app }}
version: {{ .Values.selectorLabels.version }}
{{- end -}}

{{- define "vllm-inference.image" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}

{{- define "vllm-inference.baseImage" -}}
{{- printf "%s:%s" .Values.baseImage.repository .Values.baseImage.tag -}}
{{- end -}}
