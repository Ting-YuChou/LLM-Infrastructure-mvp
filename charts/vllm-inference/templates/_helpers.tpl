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

{{- define "vllm-inference.commonLabels" -}}
helm.sh/chart: {{ include "vllm-inference.chart" . }}
app.kubernetes.io/name: {{ include "vllm-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "vllm-inference.labels" -}}
{{ include "vllm-inference.commonLabels" . }}
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

{{- define "vllm-inference.gatewayLabels" -}}
{{ include "vllm-inference.commonLabels" . }}
app.kubernetes.io/component: gateway
app: {{ .Values.gateway.selectorLabels.app }}
version: {{ .Values.gateway.selectorLabels.version }}
{{- end -}}

{{- define "vllm-inference.gatewaySelectorLabels" -}}
app: {{ .Values.gateway.selectorLabels.app }}
version: {{ .Values.gateway.selectorLabels.version }}
{{- end -}}

{{- define "vllm-inference.gatewayImage" -}}
{{- printf "%s:%s" .Values.gateway.image.repository .Values.gateway.image.tag -}}
{{- end -}}

{{- define "vllm-inference.redisLabels" -}}
{{ include "vllm-inference.commonLabels" . }}
app.kubernetes.io/component: redis
app: {{ .Values.redis.selectorLabels.app }}
{{- end -}}

{{- define "vllm-inference.redisSelectorLabels" -}}
app: {{ .Values.redis.selectorLabels.app }}
{{- end -}}

{{- define "vllm-inference.redisImage" -}}
{{- printf "%s:%s" .Values.redis.image.repository .Values.redis.image.tag -}}
{{- end -}}

{{- define "vllm-inference.redisHost" -}}
{{- if .Values.redis.externalHost -}}
{{- .Values.redis.externalHost -}}
{{- else -}}
{{- printf "%s.%s.svc.cluster.local" .Values.resourceNames.redisService (include "vllm-inference.namespace" .) -}}
{{- end -}}
{{- end -}}
