using System.Globalization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Localization;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using smart_medops_ml.Data;
using smart_medops_ml.Hubs;
using smart_medops_ml.Models;
using smart_medops_ml.Resources;
using smart_medops_ml.Services;

var builder = WebApplication.CreateBuilder(args);

var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? throw new InvalidOperationException("Connection string 'DefaultConnection' not found. Configure appsettings.json.");

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(connectionString));

builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options =>
    {
        options.Password.RequiredLength = 8;
        options.Password.RequireDigit = true;
        options.User.RequireUniqueEmail = true;
    })
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders();

builder.Services.ConfigureApplicationCookie(options =>
{
    options.LoginPath = "/Account/Login";
    options.AccessDeniedPath = "/Account/AccessDenied";
    options.SlidingExpiration = true;
});

// SharedResource lives in namespace smart_medops_ml.Resources and the RESX files
// are compiled with that base name. Keeping ResourcesPath empty avoids a duplicated
// "Resources.Resources" lookup path that makes localizer return key names.
builder.Services.AddLocalization();

builder.Services.Configure<RequestLocalizationOptions>(options =>
{
    var supported = new[] { new CultureInfo("en"), new CultureInfo("ar") };
    options.DefaultRequestCulture = new RequestCulture("en");
    options.SupportedCultures = supported;
    options.SupportedUICultures = supported;

    // Use ASP.NET Core default culture cookie to avoid split sources of truth.
    options.RequestCultureProviders.Clear();
    options.RequestCultureProviders.Add(new CookieRequestCultureProvider());
    options.RequestCultureProviders.Add(new QueryStringRequestCultureProvider
    {
        QueryStringKey = "culture",
        UIQueryStringKey = "culture",
    });
    options.RequestCultureProviders.Add(new AcceptLanguageHeaderRequestCultureProvider());
});

builder.Services.Configure<MedOpsOptions>(builder.Configuration.GetSection(MedOpsOptions.SectionName));
builder.Services.Configure<EmailSettings>(builder.Configuration.GetSection(EmailSettings.SectionName));

builder.Services.AddHttpClient<MlEmergencyScoringClient>((sp, http) =>
{
    var opt = sp.GetRequiredService<IOptions<MedOpsOptions>>().Value;
    http.Timeout = TimeSpan.FromSeconds(Math.Clamp(opt.EmergencyMlTimeoutSeconds, 1, 120));
});
builder.Services.AddHttpClient<MlCongestionPredictionClient>((sp, http) =>
{
    var opt = sp.GetRequiredService<IOptions<MedOpsOptions>>().Value;
    http.Timeout = TimeSpan.FromSeconds(Math.Clamp(opt.CongestionMlTimeoutSeconds, 1, 120));
});

builder.Services.AddScoped<DbInitializer>();
builder.Services.AddScoped<IEmailSenderService, SmtpEmailSenderService>();
builder.Services.AddScoped<EmergencyTriageService>();
builder.Services.AddScoped<IEmergencyScorer, CompositeEmergencyScorer>();
builder.Services.AddScoped<ParamedicWorkspaceService>();
builder.Services.AddScoped<CongestionPredictionService>();
builder.Services.AddHostedService<CongestionAutoRefreshWorker>();

builder.Services.AddSignalR();

builder.Services.AddControllersWithViews()
    .AddViewLocalization()
    .AddDataAnnotationsLocalization(options =>
    {
        options.DataAnnotationLocalizerProvider = (_, factory) =>
            factory.Create(typeof(SharedResource));
    });

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    try
    {
        using var scope = app.Services.CreateScope();
        await scope.ServiceProvider.GetRequiredService<DbInitializer>().SeedAsync();
    }
    catch (Exception ex)
    {
        app.Logger.LogWarning(ex, "Database seed skipped — run `dotnet ef database update` and verify SQL Server.");
    }
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseRequestLocalization();

app.UseAuthentication();
app.UseAuthorization();

app.MapHub<LiveOpsHub>("/hubs/liveops").RequireAuthorization();
app.MapControllers();
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
