package constant

type JobStatus string

const (
	JobStatusPending    JobStatus = "PENDING"
	JobStatusProcessing JobStatus = "PROCESSING"
	JobStatusFailed     JobStatus = "FAILED"
	JobStatusCompleted  JobStatus = "COMPLETED"
)

type JobType string

const (
	JobTypeTranscoder     JobType = "transcoder"
	JobTypeRecordingMerge JobType = "recording_merge"
)

type Environment string

const (
	EnvironmentProduction Environment = "production"
	EnvironmentStaging    Environment = "staging"
	EnvironmentDevelop    Environment = "develop"
)

func (e Environment) String() string {
	return string(e)
}
